#!/bin/bash
#SBATCH --job-name=rl_eval_math
#SBATCH --time=96:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=4
#SBATCH --output=logs_eval/%j.out


ml Miniconda3
ml WebProxy
ml CUDA/12.9.0  # Load CUDA module
source activate /scratch/user/vishnukunde/.conda/envs/d1

export WANDB_API_KEY=44aea80efa96b75369b009744f019926c33043f1
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}  # Set CUDA_HOME if not already set


MASTER_PORT=29411

# Arrays of tasks and generation lengths
TASKS=("math")
GEN_LENGTHS=(256)

# Multi-node settings derived from SLURM
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"
echo "SLURM_PROCID: $SLURM_PROCID"

NNODES=${SLURM_NNODES:-1}
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo "SLURM multi-node: nnodes=$NNODES master_addr=$MASTER_ADDR master_port=$MASTER_PORT"

# GPUs per node (from Slurm allocation)
NUM_GPUS=${SLURM_GPUS_ON_NODE:-2}
echo "Using $NUM_GPUS GPUs per node"

CHECKPOINT_ID="3000"
OUTPUT_DIR="eval_d1_${CHECKPOINT_ID}"

for task in "${TASKS[@]}"; do
  for gen_length in "${GEN_LENGTHS[@]}"; do
    # Set batch size based on generation length
    if [ "$gen_length" -eq 512 ]; then
      batch_size=4
    else
      batch_size=8
    fi
    
    echo "Running evaluation on $task with gen_length=$gen_length, batch_size=$batch_size across $NNODES nodes"

    srun --ntasks=$NNODES --ntasks-per-node=1 --kill-on-bad-exit=1 \
      bash -lc "torchrun \
        --nnodes $NNODES \
        --nproc_per_node $NUM_GPUS \
        --node_rank \$SLURM_PROCID \
        --master_addr $MASTER_ADDR \
        --master_port $MASTER_PORT \
        eval.py \
        --dataset $task \
        --batch_size $batch_size \
        --gen_length $gen_length \
        --output_dir $OUTPUT_DIR \
        --model_path GSAI-ML/LLaDA-8B-Instruct \
        --checkpoint_path ../diffu-grpo/checkpoints/math_base_bs12/checkpoint-${CHECKPOINT_ID}
      "
    done
done

echo "All evaluations completed!"
