#!/bin/bash
#SBATCH --job-name=math-base
#SBATCH --time=96:00:00
#SBATCH --ntasks-per-node=2
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=4
#SBATCH --output=logs/%j.out

ml Miniconda3
ml WebProxy
ml CUDA/12.9.0  # Load CUDA module
source activate /scratch/user/vishnukunde/.conda/envs/d1

export HF_HOME=/scratch/user/vishnukunde/.cache/huggingface
export WANDB_API_KEY=44aea80efa96b75369b009744f019926c33043f1

DATASET="math"
RUN_NAME=${DATASET}_base_bs12
MODEL_PATH=GSAI-ML/LLaDA-8B-Instruct
NUM_ITER=12

# Set CUDA_VISIBLE_DEVICES to ensure proper GPU assignment
# export CUDA_VISIBLE_DEVICES=0,1

accelerate launch \
    --config_file accelerate.yaml \
    --main_process_port 12346 diffu_grpo_train.py \
    --config slurm_scripts/train.yaml \
    --model_path $MODEL_PATH \
    --num_iterations $NUM_ITER \
    --dataset $DATASET \
    --run_name $RUN_NAME \
    --machine_rank $SLURM_PROCID \
    --main_process_ip $(scontrol show hostnames "$SLURM_NODELIST" | head -n1) \
    --output_dir checkpoints/$RUN_NAME