import torch
from trl.trainer.grpo_trainer import GRPOTrainer
from typing import Any, Callable, Optional, Union, Sized
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainerCallback, Trainer
from datasets import Dataset, IterableDataset
import warnings
import torch.nn.functional as F
from trl.trainer.grpo_config import GRPOConfig
from trl.extras.profiling import profiling_decorator, profiling_context
from transformers.utils import is_peft_available
from torch import nn
from trl.import_utils import is_rich_available, is_vllm_available
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.utils import (
    generate_model_card,
    get_comet_experiment_url,
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
)
import wandb

if is_peft_available():
    from peft import PeftConfig, get_peft_model
# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class TrajGRPOTrainer(GRPOTrainer):
    """
    Group Relative Policy Optimization (GRPO) Trainer for Diffusion Language Models.

    This class extends the GRPOTrainer to adapt it for masked diffusion language models,
    implementing efficient policy gradient estimation through conditional probabilities
    with masked tokens.

    Key features:
    - Random masking for improved robustness in multiple policy optimization updates
    - Efficient computation of per-token log probabilities for diffusion models
    - Specialized generation process for diffusion models with iterative denoising
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[
            Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]
        ] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[
            Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]
        ] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (
            None,
            None,
        ),
        peft_config: Optional["PeftConfig"] = None,
    ):
        # Initialize the parent class
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        trajectory_ids = inputs["trajectory_ids"]  # (per_device_train_batch_size, diffusion_steps+1, seq_len)
        eval_time_steps = inputs["eval_time_steps"]

        eval_time_step_idx = self._step % self.args.logps_eval_num_steps

        if self.args.logps_eval_mode == 'merge':
            per_token_logps = self._get_per_token_logps_merge(
                model, trajectory_ids, eval_time_steps=[eval_time_step_idx]
            ).squeeze(0) # (per_device_train_batch_size,)
        elif self.args.logps_eval_mode == 'unbiased':
            if eval_time_steps is None:
                raise ValueError(f'eval_time_steps cannot be None when logps_eval_mode is unbiased')
            per_token_logps = self._get_per_token_logps_unbiased(
                model, trajectory_ids, eval_time_steps=[eval_time_steps[eval_time_step_idx]]
            ).squeeze(0) # (per_device_train_batch_size,)
        else:
            per_token_logps = None

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"][eval_time_step_idx] # (per_device_train_batch_size,)
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            ) # (per_device_train_batch_size,)
        else:
            print(f'beta = 0.0, so no kl term is computed', flush=True)

        # Compute the loss
        advantages = inputs["advantages"]  # [per_device_train_batch_size,]

        old_per_token_logps = inputs["old_per_token_logps"][eval_time_step_idx] if self.args.logps_eval_num_steps > 1 else per_token_logps.detach()
        coef_1 = torch.exp(per_token_logps - old_per_token_logps) # [per_device_train_batch_size,]
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)
        per_token_loss1 = coef_1 * advantages # [per_device_train_batch_size,]
        per_token_loss2 = coef_2 * advantages # [per_device_train_batch_size,]
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2) # [per_device_train_batch_size,]
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl
        loss = per_token_loss.mean() # scalar
        print('step: ', self._step, 'loss: ', loss, flush=True)

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        if self.beta != 0.0:
            mean_kl = per_token_kl.mean()
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        else:
            print(f'beta = 0.0, so no kl term is logged', flush=True)

        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = is_clipped.mean()
        self._metrics[mode]["clip_ratio"].append(
            self.accelerator.gather_for_metrics(clip_ratio).mean().item()
        )

        return loss


    def add_gumbel_noise(self, logits, temperature, dtype):
        """
        The Gumbel max is a method for sampling categorical distributions.
        According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
        Thus, we use float64.
        """
        if temperature == 0.0:
            return logits  # Skip noise when temperature is 0
        logits = logits.to(dtype)
        noise = torch.rand_like(logits, dtype=dtype)
        gumbel_noise = (-torch.log(noise)) ** temperature
        return logits.exp() / gumbel_noise

    @torch.no_grad()
    def generate(
        self,
        model,
        prompt,
        steps=128,
        gen_length=128,
        block_length=128,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
        mask_id=126336,
    ):
        """generation code adopted from llada (https://github.com/ML-GSAI/LLaDA)"""
        with torch.amp.autocast(device_type='cuda', enabled=True):
            bs = prompt.shape[0]
            dtype = model.dtype
            x = torch.full((bs, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
            x[:, : prompt.shape[1]] = prompt.clone()

            prompt_index = x != mask_id

            assert gen_length % block_length == 0
            num_blocks = gen_length // block_length

            # Adjust steps if needed
            steps_per_block = max(1, steps // num_blocks)

            trajectory = []
            trajectory.append(x.clone())
            # masked_positions = []


            for num_block in range(num_blocks):
                start_idx = prompt.shape[1] + num_block * block_length
                end_idx = prompt.shape[1] + (num_block + 1) * block_length

                block_mask_index = x[:, start_idx:end_idx] == mask_id
                num_transfer_tokens = self.get_num_transfer_tokens(block_mask_index, steps_per_block)

                for i in range(steps_per_block):
                    torch.cuda.empty_cache()
                    mask_index = x == mask_id

                    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
                        with torch.cuda.amp.autocast(enabled=self.args.fp16):
                            # Handle classifier-free guidance more efficiently
                            if cfg_scale > 0.0:
                                un_x = x.clone()
                                un_x[prompt_index] = mask_id
                                x_ = torch.cat([x, un_x], dim=0)

                                # Get logits in a single forward pass
                                logits = model(x_).logits
                                logits, un_logits = torch.chunk(logits, 2, dim=0)
                                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                            else:
                                logits = model(x).logits

                            # Apply Gumbel noise for sampling
                            logits_with_noise = self.add_gumbel_noise(
                                logits, temperature=temperature, dtype=dtype
                            )
                            x0 = torch.argmax(logits_with_noise, dim=-1)
                            del logits_with_noise

                            # Handle remasking strategy
                            if remasking == "low_confidence":
                                p = F.softmax(logits.to(dtype), dim=-1)
                                x0_p = torch.squeeze(
                                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                                )
                            elif remasking == "random":
                                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                            else:
                                raise NotImplementedError(remasking)

                            # Ensure we don't process tokens beyond the current block
                            x0_p[:, end_idx:] = -np.inf

                            block_masked_positions = mask_index.clone()
                            block_masked_positions[:, end_idx:] = False

                            # masked_positions.append(block_masked_positions)

                            # Update masked tokens
                            x0 = torch.where(mask_index, x0, x)
                            confidence = torch.where(mask_index, x0_p, -np.inf)

                            # Select tokens to transfer based on confidence
                            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                            for j in range(confidence.shape[0]):
                                num_tokens = num_transfer_tokens[j, i].item()
                                if num_tokens > 0:
                                    _, select_index = torch.topk(confidence[j], k=num_tokens)
                                    transfer_index[j, select_index] = True

                            x[transfer_index] = x0[transfer_index]
                            del x0, confidence, transfer_index

                            trajectory.append(x.clone())

            # Make the trajectory a tensor
            trajectory = torch.stack(trajectory, dim=0) # (diffusion_steps+1, batch_size, seq_len)

            return trajectory


    def forward_process(self, batch, prompt_index, mask_id, seed=None):
        set_seed(seed)
        b, l = batch.shape
        t_p = torch.ones(b, device=batch.device) * self.args.p_mask_prompt

        # Create a random matrix to decide whether each prompt token is masked
        random_matrix = torch.rand((b, l), device=batch.device)

        # For prompt tokens: mask if random_matrix < t_p
        # For completion tokens: always mask
        is_mask_prompt = prompt_index & (random_matrix < t_p.unsqueeze(1))
        is_mask_completion = ~prompt_index  # all completion tokens are masked
        is_mask = is_mask_prompt | is_mask_completion

        # Create a noisy (masked) batch
        noisy_batch = torch.where(is_mask, mask_id, batch)

        # Build p_mask, the probability that each token is masked under this scheme
        #   - p_mask[i, j] = t_p[i] if it's a prompt token
        #   - p_mask[i, j] = 1      if it's a completion token
        p_mask = torch.where(
            prompt_index,
            t_p.unsqueeze(1),  # prompt token probability
            torch.ones_like(t_p).unsqueeze(1),  # completion token probability
        )

        return noisy_batch, p_mask

    def get_logits(self, model, batch, prompt_index, cfg_scale, mask_id):
        if cfg_scale > 0.0:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = mask_id
            batch = torch.cat([batch, un_batch])

        input = batch
        logits = model(input).logits

        if cfg_scale > 0.0:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
        return logits

    def get_num_transfer_tokens(self, mask_index, steps):
        """
        Precompute the number of tokens to transition at each step.
        Optimized to be more efficient.
        """
        mask_num = mask_index.sum(dim=1, keepdim=True)
        base = mask_num // steps
        remainder = mask_num % steps

        # Create tensor once and modify in-place
        num_transfer_tokens = base.expand(-1, steps).clone()

        # Handle remainder more efficiently
        if remainder.sum() > 0:
            indices = torch.arange(steps, device=mask_index.device)
            mask = indices.unsqueeze(0) < remainder
            num_transfer_tokens[mask] += 1

        return num_transfer_tokens.to(torch.int64)


   
    
    # Loop implementation of _get_per_token_logps function
    def _get_per_token_logps_unbiased(self, model, trajectory_ids, eval_time_steps=None):
        """
        Calculate per-token log probabilities.
        """

        print('Unbiased GRPO: _get_per_token_logps_unbiased', flush=True)

        # masked_positions: [diffusion_steps, batch_size, seq_len]
        _, batch_size, seq_len = trajectory_ids.size()     
        device = trajectory_ids.device
        final_state = trajectory_ids[-1, :, :] # [batch_size, seq_len]

        per_token_logps = torch.zeros(len(eval_time_steps), batch_size, device=device)  # [len(sub_steps), batch_size]

        for i in range(len(eval_time_steps)):

            curr_state = trajectory_ids[eval_time_steps[i], :,  :] # [batch_size, seq_len]
            next_state = trajectory_ids[eval_time_steps[i] + 1, :, :] # [batch_size, seq_len]             

            if self.args.pred_state == 'next':
                positions = curr_state != next_state # [batch_size, seq_len]
                targets = next_state[positions] # [positions.sum()]
            elif self.args.pred_state == 'final':
                positions = curr_state != final_state # [batch_size, seq_len]
                targets = final_state[positions] # [positions.sum()]
            else:
                raise ValueError(f'Invalid pred_state: {self.args.pred_state}')

            pred_logits_all = model(curr_state).logits # [batch_size, seq_len, vocab_size]
            pred_logits = pred_logits_all[positions] # [positions.sum(), vocab_size]

            # Compute per-token cross-entropy losses
            per_token_losses = F.cross_entropy(pred_logits, targets, reduction="none")  # [positions.sum()]

            # Aggregate per-token losses back to per-batch losses
            batch_indices_all = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, seq_len)  # [batch_size, seq_len]
            batch_indices = batch_indices_all[positions]  # [positions.sum()]

            # print(f'batch_indices (positions.sum(),) = ({batch_indices.shape})', flush=True)
            # print(f'per_token_losses (positions.sum(),) = ({per_token_losses.shape})', flush=True)

            # Sum the losses
            summed_losses = torch.zeros(batch_size, device=device, dtype=per_token_losses.dtype).scatter_add_(
                0, batch_indices, per_token_losses
            )  # [batch_size] summed losses for each batch where the token is unmasked
            # print(f'summed_losses (batch_size,) = ({summed_losses.shape})', flush=True)
            # Count the tokens per batch
            counts = torch.zeros(batch_size, device=device, dtype=per_token_losses.dtype).scatter_add_(
                0, batch_indices, torch.ones_like(per_token_losses)  # [batch_size] counts for each batch where the token is unmasked
            )
            # print(f'counts (batch_size,) = ({counts.shape})', flush=True)
            # Divide sum by count to get mean
            per_token_logps[i, :] = -summed_losses / counts.clamp(min=1.0)  # clamp to avoid division by zero

        torch.cuda.empty_cache()
        return per_token_logps  # [len(sub_steps), batch_size]


    # Loop implementation of _get_per_token_logps function
    def _get_per_token_logps_merge(self, model, trajectory_ids, eval_time_steps=None):
        """
        Calculate per-token log probabilities.
        """

        print('Merge GRPO: _get_per_token_logps_merge', flush=True)

        num_sub_steps = self.args.logps_eval_num_steps
        pred_state = self.args.pred_state

        # masked_positions: [diffusion_steps, batch_size, seq_len]
        diffusion_steps_with_init, batch_size, seq_len = trajectory_ids.size()      
        device = trajectory_ids.device
        diffusion_steps = diffusion_steps_with_init - 1
        final_state = trajectory_ids[-1, :, :] # [batch_size, seq_len]

        leap = diffusion_steps // num_sub_steps

        # Subsample the trajectory and masked_positions
        sub_trajectory_ids = trajectory_ids[:diffusion_steps:leap, :, :]
        sub_trajectory_ids = torch.cat([sub_trajectory_ids, final_state.unsqueeze(0)], dim=0) # Add the final state of the trajectory

        if eval_time_steps is None:
            eval_time_steps = range(0, num_sub_steps, 1)

        per_token_logps = torch.zeros(len(eval_time_steps), batch_size, device=device)  # [len(eval_time_steps), batch_size]

        for i in range(len(eval_time_steps)):

            curr_state = sub_trajectory_ids[eval_time_steps[i], :,  :] # [batch_size, seq_len]
            next_state = sub_trajectory_ids[eval_time_steps[i] + 1, :, :] # [batch_size, seq_len]             

            if pred_state == 'next':
                positions = curr_state != next_state # [batch_size, seq_len]
                targets = next_state[positions] # [positions.sum()]
            elif pred_state == 'final':
                positions = curr_state != final_state # [batch_size, seq_len]
                targets = final_state[positions] # [positions.sum()]
            else:
                raise ValueError(f'Invalid pred_state: {pred_state}')

            pred_logits_all = model(curr_state).logits # [batch_size, seq_len, vocab_size]
            pred_logits = pred_logits_all[positions] # [positions.sum(), vocab_size]

            # Compute per-token cross-entropy losses
            per_token_losses = F.cross_entropy(pred_logits, targets, reduction="none")  # [positions.sum()]

            # Aggregate per-token losses back to per-batch losses
            batch_indices_all = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, seq_len)  # [batch_size, seq_len]
            batch_indices = batch_indices_all[positions]  # [positions.sum()]

            # print(f'batch_indices (positions.sum(),) = ({batch_indices.shape})', flush=True)
            # print(f'per_token_losses (positions.sum(),) = ({per_token_losses.shape})', flush=True)

            # Sum the losses
            summed_losses = torch.zeros(batch_size, device=device, dtype=per_token_losses.dtype).scatter_add_(
                0, batch_indices, per_token_losses
            )  # [batch_size] summed losses for each batch where the token is unmasked
            # print(f'summed_losses (batch_size,) = ({summed_losses.shape})', flush=True)
            # Count the tokens per batch
            counts = torch.zeros(batch_size, device=device, dtype=per_token_losses.dtype).scatter_add_(
                0, batch_indices, torch.ones_like(per_token_losses)  # [batch_size] counts for each batch where the token is unmasked
            )
            # print(f'counts (batch_size,) = ({counts.shape})', flush=True)
            # Divide sum by count to get mean
            per_token_logps[i, :] = -summed_losses / counts.clamp(min=1.0)  # clamp to avoid division by zero

        torch.cuda.empty_cache()
        return per_token_logps  # [len(sub_steps), batch_size]


    # def _prepare_inputs(
    #     self, inputs: dict[str, Union[torch.Tensor, Any]]
    # ) -> dict[str, Union[torch.Tensor, Any]]:
    #     mode = "eval" if self.control.should_evaluate else "train"
    #     if mode == "train":
    #         if self.state.global_step % self.num_iterations == 0:
    #             inputs = self._generate_and_score_completions(inputs)
    #             self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = inputs
    #         else:
    #             inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
    #         self._step += 1
    #     else:
    #         # In evaluation, we don't reuse completions across multiple updates, so we don't need to buffer inputs.
    #         inputs = self._generate_and_score_completions(inputs)
    #     return inputs


    def _prepare_inputs(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        mode = "eval" if self.control.should_evaluate else "train"
        if mode == "train":
            if self._step % self.args.logps_eval_num_steps == 0:
                inputs = self._generate_and_score_completions(inputs)
                self._cached_inputs = inputs
            else:
                inputs = self._cached_inputs
            self._step += 1
        return inputs

    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device

        prompts = [x["prompt"] for x in inputs]
        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs
        ]
        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Configuration for the diffusion generation
        gen_length = self.args.max_completion_length
        block_length = self.args.block_length
        steps = self.args.diffusion_steps
        temperature = self.args.temperature or 0.0
        cfg_scale = self.args.cfg_scale

        with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
            generation_batch_size = self.args.generation_batch_size
            trajectory_ids_all = []
            # Process in batches
            for i in range(0, prompt_ids.size(0), generation_batch_size):
                end_idx = min(i + generation_batch_size, prompt_ids.size(0))
                batch_prompt_ids = prompt_ids[i:end_idx]
                batch_prompt_mask = prompt_mask[i:end_idx]
                # WARNING: Attention masks are not currently used during generation.
                # This works fine as we set num_generations == per_device_train_batch_size (no padding tokens created) in our config, but may cause
                # unintended attention to padding tokens when num_generations is smaller.
                # As currently we find Llada's modeling file does not handle attention mask. We will address this in future update soon.

                # Modify the generate function to return the whole trajectory instead of just the completion tokens
                batch_trajectory_ids = self.generate(
                    model=unwrapped_model,
                    prompt=batch_prompt_ids,
                    steps=steps,
                    gen_length=gen_length,
                    block_length=block_length,
                    temperature=temperature,
                    cfg_scale=cfg_scale,
                    remasking=self.args.remasking,
                    mask_id=self.args.mask_id,
                )  # (diffusion_steps+1, generation_batch_size, seq_len)

                # Permute the trajectory to (generation_batch_size, diffusion_steps+1, seq_len) and add to the list
                trajectory_ids_all.append(batch_trajectory_ids.permute(1, 0, 2))

                del batch_prompt_ids, batch_prompt_mask, batch_trajectory_ids
                torch.cuda.empty_cache()

            # (generation_batch_size, diffusion_steps+1, seq_len) -> (num_batches * generation_batch_size, diffusion_steps+1, seq_len) -> (diffusion_steps+1, per_device_train_batch_size, seq_len)
            trajectory_ids = torch.cat(trajectory_ids_all, dim=0).permute(1, 0, 2) # (diffusion_steps+1, per_device_train_batch_size, seq_len)

        # Compute prompt length and extract completion ids
        prompt_length = prompt_ids.size(-1)
        prompt_ids = trajectory_ids[:, :, :prompt_length]
        response_trajectory_ids = trajectory_ids[:, :, prompt_length:]
        completion_ids = response_trajectory_ids[-1, :, :]  # final state of the trajectory [batch_size, completion_length]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(-1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(-1), device=device).expand(is_eos.size(0), -1) # (batch_size, completion_length)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        logits_to_keep = completion_ids.size(
            1
        )  # we only need to compute the logits for the completion tokens

        diffusion_steps = trajectory_ids.size(0) - 1

        if self.args.logps_eval_mode == 'unbiased':
            if self.args.logps_eval_time_steps_mode == 'random':
                eval_time_steps = torch.randint(0, diffusion_steps-1, (self.args.logps_eval_num_steps-1,)).to(device)
                # Add the final time step to eval_time_steps
                eval_time_steps = torch.cat([eval_time_steps, torch.full((1,), diffusion_steps-1, device=device)])
                print(f'eval_time_steps = {eval_time_steps}', flush=True)
            elif self.args.logps_eval_time_steps_mode == 'uniform':
                eval_time_steps = torch.linspace(0, diffusion_steps-2, self.args.logps_eval_num_steps-1).long().to(device)
                # Add the final time step to eval_time_steps
                eval_time_steps = torch.cat([eval_time_steps, torch.full((1,), diffusion_steps-1, device=device)])
                print(f'eval_time_steps = {eval_time_steps}', flush=True)
            else:
                eval_time_steps = None
        

        all_old_per_token_logps = []
        all_ref_per_token_logps = []
        with torch.no_grad():
            if self.args.logps_eval_num_steps > 1:
                if self.args.logps_eval_mode == 'merge':
                    all_old_per_token_logps = self._get_per_token_logps_merge(
                        self.model, trajectory_ids
                    )
                elif self.args.logps_eval_mode == 'unbiased':
                    all_old_per_token_logps = self._get_per_token_logps_unbiased(
                        self.model, trajectory_ids, eval_time_steps=eval_time_steps)
            else:
                all_old_per_token_logps = None
            

            if self.beta == 0.0:
                all_ref_per_token_logps = None
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    if self.args.logps_eval_mode == 'merge':
                        all_ref_per_token_logps = self._get_per_token_logps_merge(
                            self.model, trajectory_ids
                        )
                    elif self.args.logps_eval_mode == 'unbiased':
                        all_ref_per_token_logps = self._get_per_token_logps_unbiased(
                            self.model, trajectory_ids, eval_time_steps=eval_time_steps)
                    else:
                        all_ref_per_token_logps = None

        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        # for i in range(len(completions)):
        #     print(f'machine id: {self.accelerator.process_index}, device: {device}, completion {i}: {completions[i]} \n', flush=True)

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(
                reward_func, nn.Module
            ):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = f"reward {reward_func.config._name_or_path.split('/')[-1]}"
            else:
                reward_func_name = reward_func.__name__
            with profiling_context(self, reward_func_name):

                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                output_reward_func = reward_func(
                    prompts=prompts,
                    completions=completions,
                    step=self._step,
                    run_name=self.args.output_dir,
                    **reward_kwargs,
                )
                # Convert None values to NaN
                output_reward_func = [
                    reward if reward is not None else torch.nan for reward in output_reward_func
                ]

                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

        print(f'rewards_per_func (batch_size, num_reward_funcs) = ({rewards_per_func.shape})', flush=True)


        rewards_per_func = gather(rewards_per_func)
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        print(f'rewards (batch_size,) = ({rewards.shape})', flush=True)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards

        # print(f'advantages (batch_size,) = ({advantages.shape})', flush=True)
        # Count prompts with zero std deviation
        zero_std_count = (std_grouped_rewards < 1e-6).sum().item()  # Using a small threshold
        total_prompts = std_grouped_rewards.size(0)
        zero_std_ratio = zero_std_count / total_prompts if total_prompts > 0 else 0.0

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics[mode]["completion_length"].append(completion_length)
        self._metrics[mode]["zero_std_ratio"].append(zero_std_ratio)

        # Calculate mean reward per function, but only for samples where the function was applied
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(
                reward_func, nn.Module
            ):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            # Only calculate mean for samples where this reward function was applied (non-NaN values)
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}"].append(mean_rewards)
        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            prompts_to_log = gather_object(prompts_text)
            completions_to_log = gather_object(completions_text)
            rewards_to_log = rewards.tolist()

            if self.accelerator.is_main_process:
                if is_rich_available():
                    print_prompt_completions_sample(
                        prompts_to_log,
                        completions_to_log,
                        rewards_to_log,
                        self.state.global_step,
                    )
                if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                    import pandas as pd

                    # For logging
                    table = {
                        "step": [str(self.state.global_step)] * len(rewards),
                        "prompt": prompts_to_log,
                        "completion": completions_to_log,
                        "reward": rewards.tolist(),
                    }
                    df = pd.DataFrame(table)
                    wandb.log({"completions": wandb.Table(dataframe=df)})

        return {
            "trajectory_ids": trajectory_ids,
            "completion_ids": completion_ids,
            "eval_time_steps": eval_time_steps,
            "old_per_token_logps": all_old_per_token_logps,
            "ref_per_token_logps": all_ref_per_token_logps,
            "advantages": advantages,
        }

    def _get_train_sampler(self, dataset):
        """
        Override the parent method to handle the dataset parameter correctly.
        This fixes the TypeError where the method was being called with 2 arguments
        but only expected 1.
        """
        return super()._get_train_sampler()
