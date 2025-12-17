import torch
import wandb
import os
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from trl import TrlParser, ModelConfig
from peft import LoraConfig

# Custom imports
from unbiased_entropy_grpo_trainer import TrajGRPOTrainer
from traj_grpo_config import TrajGRPOConfig
from reward_func import (
    xmlcount_reward_func,
    soft_format_reward_func,
    strict_format_reward_func,
    int_reward_func,
    correctness_reward_func,
    countdown_reward_func,
    correctness_reward_func_math,
    sudoku_reward_func,
    boxed_and_answer_tags_format_reward,
    reward_len,
)
from data_utils import (
    get_gsm8k_questions,
    get_countdown_questions,
    get_sudoku_questions,
    set_random_seed,
    get_math_questions,
)


def main(grpo_config, model_config):

    # Initialize wandb if WANDB_ID is set (for resuming)
    # Only initialize on rank 0 to avoid multiple wandb instances in distributed training
    # Read global_step from checkpoint if resuming to ensure correct step logging
    rank_str = os.environ.get("RANK")
    rank = int(rank_str) if rank_str is not None else 0
    wandb_id = os.environ.get("WANDB_ID")
    wandb_resume = os.environ.get("WANDB_RESUME", "allow")
    
    # Get global_step from checkpoint if resuming
    resume_step = None
    if grpo_config.resume_from_checkpoint and rank == 0:
        checkpoint_path = Path(grpo_config.resume_from_checkpoint)
        training_state_file = checkpoint_path / "training_state.json"
        if training_state_file.exists():
            try:
                with open(training_state_file, 'r') as f:
                    training_state = json.load(f)
                    resume_step = training_state.get("global_step", None)
                    print(f"Found checkpoint at step {resume_step} from training_state.json", flush=True)
            except Exception as e:
                print(f"Warning: Could not read training_state.json: {e}", flush=True)
        
        # Fallback: try to extract step from checkpoint directory name (e.g., checkpoint-4500)
        if resume_step is None:
            checkpoint_name = checkpoint_path.name
            if checkpoint_name.startswith("checkpoint-"):
                try:
                    resume_step = int(checkpoint_name.split("-")[1])
                    print(f"Extracted checkpoint step {resume_step} from directory name", flush=True)
                except (ValueError, IndexError):
                    print(f"Warning: Could not extract step from checkpoint name: {checkpoint_name}", flush=True)
    
    if rank == 0 and wandb_id and grpo_config.report_to and "wandb" in grpo_config.report_to and wandb.run is None:
        print(f"Initializing wandb with run_id={wandb_id}, resume={wandb_resume}, checkpoint_step={resume_step}", flush=True)
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "huggingface"),
            id=wandb_id,
            resume=wandb_resume,
        )
        # Set the step from checkpoint if resuming
        if resume_step is not None and wandb.run is not None:
            wandb.run.step = resume_step
            print(f"Set wandb step to {resume_step} from checkpoint", flush=True)

    # Set seed for reproducibility
    set_random_seed(grpo_config.seed)

    # Load dataset based on configuration
    if grpo_config.dataset == "gsm8k":
        dataset = get_gsm8k_questions("train")
        reward_functions = [
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ]
    elif grpo_config.dataset == "countdown":
        dataset = get_countdown_questions("train")
        reward_functions = [countdown_reward_func]
    elif grpo_config.dataset == "sudoku":
        dataset = get_sudoku_questions()
        reward_functions = [sudoku_reward_func]
    elif grpo_config.dataset == "math":
        dataset = get_math_questions("train")
        reward_functions = [
            correctness_reward_func_math,
            boxed_and_answer_tags_format_reward,
        ]

    # Shuffle dataset with fixed seed for reproducibility
    dataset = dataset.shuffle(seed=grpo_config.seed)

    # Split dataset if needed
    if grpo_config.dataset in ["countdown", "sudoku"]:
        train_set = dataset.select(range(0, len(dataset) - 500))  # Leave last 500 for evaluation
    else:
        train_set = dataset

    print('Length of train set:', len(train_set), flush=True)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4 bit quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print('Loading model... from', f'{grpo_config.model_path}', flush=True)

    # Load model and tokenizer
    model = AutoModel.from_pretrained(
        grpo_config.model_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        quantization_config=bnb_config,
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(grpo_config.model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False

    # Configure LoRA for parameter-efficient fine-tuning
    peft_config = LoraConfig(
        r=model_config.lora_r,
        lora_alpha=model_config.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=model_config.lora_dropout,
    )
    # Initialize and run trainer
    trainer = TrajGRPOTrainer(
        args=grpo_config,
        model=model,
        peft_config=peft_config,
        reward_funcs=reward_functions,
        train_dataset=train_set,
    )

    trainer.train()


if __name__ == "__main__":
    parser = TrlParser((TrajGRPOConfig, ModelConfig))
    grpo_config, model_config = parser.parse_args_and_config()
    main(grpo_config=grpo_config, model_config=model_config)
