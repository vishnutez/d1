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

        completion_mask = inputs["completion_mask"]
        prompt_trajectory_ids = inputs["prompt_trajectory_ids"]  # (batch_size, traj_len, seq_len)
        per_token_logps = self._get_per_token_logps(model, prompt_trajectory_ids) # (batch_size, subsampled_steps)
        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"] # (batch_size, subsampled_steps)
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            ) # (batch_size, subsampled_steps)

        # TODO: Implement num_iterations > 1

        # Compute the loss
        advantages = inputs["advantages"]  # [batch_size,]
        old_per_token_logps = (
            inputs["old_per_token_logps"] if self.args.num_iterations > 1 else per_token_logps.detach()
        ) # [batch_size, subsampled_steps]

        coef_1 = torch.exp(per_token_logps - old_per_token_logps) # [batch_size, subsampled_steps]
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1) # [batch_size, subsampled_steps]
        per_token_loss2 = coef_2 * advantages.unsqueeze(1) # [batch_size, subsampled_steps]
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2) # [batch_size, subsampled_steps]
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl
        loss = per_token_loss.mean()
        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        if self.beta != 0.0:
            mean_kl = per_token_kl.mean()
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

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

            # make the trajectory a tensor
            trajectory = torch.stack(trajectory, dim=0)
            # print('In generate func, trajectory shape: ', trajectory.shape, flush=True)
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

    

    # def _get_per_token_logps(self, model, input_ids, leap=16):
    #     """
    #     Calculate per-token log probabilities.
    #     """

    #     # input_ids: [batch_size, traj_len, seq_len]
    #     batch_size, traj_len, seq_len = input_ids.size()

    #     print(f'input_ids (batch_size, traj_len, seq_len) = ({input_ids.shape})', flush=True)


    #     diffusion_steps = traj_len - 1
    #     device = input_ids.device
    #     dtype = input_ids.dtype

    #     # subsampled input_ids
    #     subsampled_input_ids = input_ids[:, ::leap, :] # [batch_size, traj_len/leap, seq_len]

    #     targets = subsampled_input_ids[:, 1:, :] # [batch_size, traj_len/leap-1, seq_len] (next state)
    #     inputs = subsampled_input_ids[:, :-1, :] # [batch_size, traj_len/leap-1, seq_len] (current state)


    #     logits = model(inputs).logits # [batch_size, traj_len/leap-1, seq_len, vocab_size]

    #     print(f'subsampled_input_ids (batch_size, traj_len/leap, seq_len) = ({subsampled_input_ids.shape})', flush=True)


    #     per_token_logps = torch.zeros(batch_size, diffusion_steps // leap, device=device, dtype=dtype)



    #     print(f'per_token_logps (batch_size, diffusion_steps/leap) = ({per_token_logps.shape})', flush=True)

    #     for step in range(0, diffusion_steps, leap):
    #         x_curr = input_ids[:, step, :]  # [batch_size, seq_len]
    #         x_next = input_ids[:, step + leap, :]  # [batch_size, seq_len]

    #         # find the locations when x_next is different from x_curr
    #         unmasked_positions = x_next != x_curr  # [batch_size, seq_len]

    #         print(f'unmasked_positions shape = {unmasked_positions.shape}', flush=True)

    #         # get the logits for the next state given the current state
    #         next_pred_logits = model(x_curr).logits  # [batch_size, seq_len, vocab_size]

    #         print(f'next_pred_logits shape = {next_pred_logits.shape}', flush=True)

    #         # Select logits where unmasked_positions is True
    #         # unmasked_positions: [batch_size, seq_len] (boolean)
    #         # next_pred_logits: [batch_size, seq_len, vocab_size]
    #         # Result: [num_total_unmasked, vocab_size] where num_total_unmasked is total True values
    #         unmasked_logits = next_pred_logits[unmasked_positions]  # [num_total_unmasked, vocab_size]

    #         # Select target tokens where unmasked_positions is True
    #         unmasked_targets = x_next[unmasked_positions]  # [num_total_unmasked]

    #         print(f'unmasked_logits shape = {unmasked_logits.shape}', flush=True)
    #         print(f'unmasked_targets shape = {unmasked_targets.shape}', flush=True)

    #         # Compute per-token cross-entropy losses
    #         per_token_losses = F.cross_entropy(unmasked_logits, unmasked_targets, reduction="none")  # [num_total_unmasked]

    #         # Aggregate per-token losses back to per-batch losses
    #         # Create batch indices tensor to track which batch each unmasked token belongs to
    #         batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, seq_len)  # [batch_size, seq_len]
    #         batch_indices_unmasked = batch_indices[unmasked_positions]  # [num_total_unmasked]

    #         # Sum losses per batch using scatter_add
    #         step_idx = step // leap
    #         per_token_logps[:, step_idx] = torch.zeros(batch_size, device=device, dtype=per_token_losses.dtype).scatter_add_(
    #             0, batch_indices_unmasked, per_token_losses
    #         )

    #     print(f'per_token_logps (batch_size, diffusion_steps/leap) = ({per_token_logps.shape})', flush=True)

    #     torch.cuda.empty_cache()
    #     return per_token_logps  # [batch_size, diffusion_steps/leap]


    # Loop implementation of _get_per_token_logps function

    def _get_per_token_logps(self, model, input_ids, leap=32, answer_only=True):
        """
        Calculate per-token log probabilities.
        """

        # input_ids: [batch_size, traj_len, seq_len]
        batch_size, traj_len, seq_len = input_ids.size()

        answer_ids = input_ids[:, -1, :] # [batch_size, seq_len]

        # print(f'input_ids (batch_size, traj_len, seq_len) = ({input_ids.shape})', flush=True)


        diffusion_steps = traj_len - 1
        device = input_ids.device
        per_token_logps = torch.zeros(batch_size, diffusion_steps // leap, device=device)

        # print(f'per_token_logps (batch_size, diffusion_steps/leap) = ({per_token_logps.shape})', flush=True)

        for step in range(0, diffusion_steps, leap):
            x_curr = input_ids[:, step, :]  # [batch_size, seq_len]

            if answer_only:
                positions = answer_ids != x_curr  # [batch_size, seq_len]
            else:
                x_next = input_ids[:, step + leap, :]  # [batch_size, seq_len]
                positions = x_next != x_curr  # [batch_size, seq_len]

            # get the logits for the next state given the current state
            pred_logits_all = model(x_curr).logits  # [batch_size, seq_len, vocab_size]
            pred_logits = pred_logits_all[positions]  # [num_total_unmasked, vocab_size]

            if answer_only:
                targets = answer_ids[positions]  # [num_total_unmasked]
            else:
                targets = x_next[positions]  # [num_total_unmasked]

            # Compute per-token cross-entropy losses
            per_token_losses = F.cross_entropy(pred_logits, targets, reduction="none")  # [num_total_unmasked]

            # Aggregate per-token losses back to per-batch losses
            # Create batch indices tensor to track which batch each unmasked token belongs to
            batch_indices_all = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, seq_len)  # [batch_size, seq_len]
            batch_indices = batch_indices_all[positions]  # [num_total_unmasked]

            # Mean losses per batch using scatter_add for sum and counts
            step_idx = step // leap
            # Sum the losses
            summed_losses = torch.zeros(batch_size, device=device, dtype=per_token_losses.dtype).scatter_add_(
                0, batch_indices, per_token_losses
            )
            # Count the tokens per batch
            counts = torch.zeros(batch_size, device=device, dtype=per_token_losses.dtype).scatter_add_(
                0, batch_indices, torch.ones_like(per_token_losses)
            )
            # Divide sum by count to get mean
            per_token_logps[:, step_idx] = summed_losses / counts.clamp(min=1.0)  # clamp to avoid division by zero

        # print(f'per_token_logps (batch_size, diffusion_steps/leap) = ({per_token_logps.shape})', flush=True)

        torch.cuda.empty_cache()
        return per_token_logps  # [batch_size, diffusion_steps/leap]



    def _prepare_inputs(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        mode = "eval" if self.control.should_evaluate else "train"
        if mode == "train":
            if self.state.global_step % self.num_iterations == 0:
                inputs = self._generate_and_score_completions(inputs)
                self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = inputs
            else:
                inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
            self._step += 1
        else:
            # In evaluation, we don't reuse completions across multiple updates, so we don't need to buffer inputs.
            inputs = self._generate_and_score_completions(inputs)
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
            prompt_trajectory_ids_all = []
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
                batch_prompt_trajectory_ids = self.generate(
                    model=unwrapped_model,
                    prompt=batch_prompt_ids,
                    steps=steps,
                    gen_length=gen_length,
                    block_length=block_length,
                    temperature=temperature,
                    cfg_scale=cfg_scale,
                    remasking=self.args.remasking,
                    mask_id=self.args.mask_id,
                )  # (traj_len=diffusion_steps+1, batch_size, seq_len)  1 for the prompt, diffusion_steps for the completion

                # swap dimensions of batch_prompt_trajectory_ids to 
                batch_prompt_trajectory_ids = batch_prompt_trajectory_ids.permute(1, 0, 2) # (batch_size, traj_len, seq_len)

                prompt_trajectory_ids_all.append(batch_prompt_trajectory_ids)

                del batch_prompt_ids, batch_prompt_mask, batch_prompt_trajectory_ids
                torch.cuda.empty_cache()

            prompt_trajectory_ids = torch.cat(prompt_trajectory_ids_all, dim=0) # (batch_size, traj_len, seq_len)

        # print(f'prompt_trajectory_ids (after trajectory roll-out from old_policy) (batch_size, traj_len, seq_len) = ({prompt_trajectory_ids.shape})', flush=True)

        # Compute prompt length and extract completion ids
        prompt_length = prompt_ids.size(-1)
        print(f'prompt_length = {prompt_length}', flush=True)
        prompt_ids = prompt_trajectory_ids[:, :, :prompt_length]
        trajectory_ids = prompt_trajectory_ids[:, :, prompt_length:]
        completion_ids = trajectory_ids[:, -1, :]  # final state of the trajectory [batch_size, completion_length]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(-1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(-1), device=device).expand(is_eos.size(0), -1) # (batch_size, completion_length)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        logits_to_keep = completion_ids.size(
            1
        )  # we only need to compute the logits for the completion tokens


        all_old_per_token_logps = []
        all_ref_per_token_logps = []
        with torch.no_grad():

            # print(f'Evaluating old per-token logps for the trajectory', flush=True)

            # get logps for the prompt trajectory TODO: these can be saved during generation and need not be computed again
            if self.args.num_iterations > 1:
                all_old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_trajectory_ids
                )
            else:
                all_old_per_token_logps = None
            
            # print(f'all_old_per_token_logps (batch_size, diffusion_steps) = ({all_old_per_token_logps.shape})', flush=True)
            # print('Done evaluating old per-token logps for the trajectory', flush=True)

            if self.beta == 0.0:
                all_ref_per_token_logps = None
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    # print(f'Evaluating ref per-token logps for the trajectory', flush=True)
                    all_ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_trajectory_ids
                    )
                    # print(f'all_ref_per_token_logps (batch_size, diffusion_steps) = ({all_ref_per_token_logps.shape})', flush=True)
                    # print('Done evaluating ref per-token logps for the trajectory', flush=True)

        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

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

        rewards_per_func = gather(rewards_per_func)
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

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
            "prompt_trajectory_ids": prompt_trajectory_ids,
            "trajectory_ids": trajectory_ids,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
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
