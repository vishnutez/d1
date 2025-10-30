#!/bin/bash
#SBATCH --job-name=d1_gsm8k_bs4
#SBATCH --time=96:00:00
#SBATCH --ntasks-per-node=2
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=4
#SBATCH --output=logs/%x_%j.out

# ----------------------------
# User-configurable parameters
# ----------------------------
MASTER_PORT=12346
CFGDIR="./accel_cfg"
PRECISION="bf16"                  # bf16 | fp16 | no

# DeepSpeed config options
DS_ZERO_STAGE=2
DS_OVERLAP_COMM=true
DS_GRAD_CLIP=1
DS_OFFLOAD_OPT="none"
DS_OFFLOAD_PARAM="none"
DS_ZERO3_INIT=false

DATASET="gsm8k"
RUN_NAME=${DATASET}_base_bs4
MODEL_PATH=GSAI-ML/LLaDA-8B-Instruct
NUM_ITER=4
GEN_BATCH_SIZE=4

# Your diffu_grpo_train.py args (editable)
TRAIN_SCRIPT="diffu_grpo_train.py"
TRAIN_ARGS_BASE="--config slurm_scripts/train.yaml"
TRAIN_ARGS_EXTRA="--model_path ${MODEL_PATH} \
                  --num_iterations ${NUM_ITER} \
                  --dataset ${DATASET} \
                  --run_name ${RUN_NAME} \
                  --output_dir checkpoints/${RUN_NAME} \
                  --generation_batch_size ${GEN_BATCH_SIZE}"

# ----------------------------
# Environment setup
# ----------------------------

ml Miniconda3
ml WebProxy
ml CUDA/12.9.0  # Load CUDA module
source activate /scratch/user/vishnukunde/.conda/envs/d1

export WANDB_API_KEY=44aea80efa96b75369b009744f019926c33043f1
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}  # Set CUDA_HOME if not already set

mkdir -p logs "$CFGDIR"

# ----------------------------
# Cluster topology
# ----------------------------
NUM_MACHINES=${SLURM_NNODES:?SLURM_NNODES not set}
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-1}}
MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
TOTAL_PROCS=$((NUM_MACHINES * GPUS_PER_NODE))

echo "MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT"
echo "NUM_MACHINES=$NUM_MACHINES  GPUS_PER_NODE=$GPUS_PER_NODE  TOTAL_PROCS=$TOTAL_PROCS"

# ----------------------------
# Generate per-machine Accelerate configs
# ----------------------------
for RANK in $(seq 0 $((NUM_MACHINES - 1))); do
cat > "${CFGDIR}/accelerate_machine_${RANK}.yaml" <<EOF
compute_environment: LOCAL_MACHINE
debug: false

distributed_type: DEEPSPEED
main_training_function: main
mixed_precision: '${PRECISION}'
downcast_bf16: 'auto'

rdzv_backend: static
same_network: true
main_process_ip: ${MASTER_ADDR}
main_process_port: ${MASTER_PORT}

num_machines: ${NUM_MACHINES}
num_processes: ${TOTAL_PROCS}
machine_rank: ${RANK}
gpu_ids: all
use_cpu: false

deepspeed_config:
  deepspeed_multinode_launcher: standard
  zero_stage: ${DS_ZERO_STAGE}
  zero3_init_flag: ${DS_ZERO3_INIT}
  offload_optimizer_device: ${DS_OFFLOAD_OPT}
  offload_param_device: ${DS_OFFLOAD_PARAM}
  overlap_comm: ${DS_OVERLAP_COMM}
  gradient_clip: ${DS_GRAD_CLIP}

tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
EOF
done

# ----------------------------
# Launch with srun (one task per node)
# ----------------------------
export MASTER_ADDR="$MASTER_ADDR"
export MASTER_PORT="$MASTER_PORT"

echo "Launching training with srun (Accelerate + DeepSpeed)..."
srun \
  --ntasks="$NUM_MACHINES" \
  --nodes="$NUM_MACHINES" \
  --ntasks-per-node=1 \
  bash -lc '
    export WANDB_API_KEY='"$WANDB_API_KEY"'
    export CUDA_HOME='"$CUDA_HOME"'
    ID=${SLURM_PROCID}
    echo ">>> Node $(hostname) starting machine_rank=${ID}"
    accelerate launch \
      --config_file '"$CFGDIR"'/accelerate_machine_${ID}.yaml \
      --main_process_port '"$MASTER_PORT"' \
      '"$TRAIN_SCRIPT"' \
      '"$TRAIN_ARGS_BASE"' \
      '"$TRAIN_ARGS_EXTRA"'
  '

echo "All nodes completed."
