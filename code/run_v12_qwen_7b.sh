#!/bin/bash
#SBATCH --job-name=grpo_aco_v12_qwen_7b
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --time=196:00:00
#SBATCH --partition=

source 
conda activate tool

cd 

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export MASTER_ADDR=localhost
export MASTER_PORT=$(shuf -i 20000-60000 -n 1)
echo "DDP Master Addr: $MASTER_ADDR"
echo "DDP Master Port: $MASTER_PORT"

mkdir -p logs


torchrun --nproc_per_node=2 \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         train_grpo_aco_v12_qwen_7b.py

EXIT_CODE=$?

echo ""
echo "=============================================================================="
echo "v12.1 Qwen-7B (LLM Simulator) Training Completed"
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=============================================================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Training succeeded! View results:"
    echo ""
    echo "  [Main Results]"
    echo "  cat checkpoints_v12_qwen_7b_llm_sim/results.json"
    echo ""
    echo "  [Training Records (for plotting)]"
    echo "  - Training step log (JSON): checkpoints_v12_qwen_7b_llm_sim/training_step_log.json"
    echo "  - Training step log (CSV):  checkpoints_v12_qwen_7b_llm_sim/training_step_log.csv"
    echo "  - Pheromone evolution:      checkpoints_v12_qwen_7b_llm_sim/pheromone_evolution.json"
    echo "  - Loss curve data:          checkpoints_v12_qwen_7b_llm_sim/loss_curve.json"
    echo "  - Accuracy curve data:      checkpoints_v12_qwen_7b_llm_sim/accuracy_curve.json"
    echo "  - Reward curve data:        checkpoints_v12_qwen_7b_llm_sim/reward_curve.json"
    echo ""
    echo "  [Model Checkpoints]"
    echo "  - SL best:       checkpoints_v12_qwen_7b_llm_sim/sl_best.pt"
    echo "  - Mixed best:    checkpoints_v12_qwen_7b_llm_sim/mixed_best.pt"
    echo "  - RL best:       checkpoints_v12_qwen_7b_llm_sim/rl_best.pt"
    echo "  - Final model:   checkpoints_v12_qwen_7b_llm_sim/policy_final.pt"
    echo ""
    echo "  [Detailed Predictions (for manual review)]"
    echo "  cat checkpoints_v12_qwen_7b_llm_sim/predictions_detailed.json | head -200"
    echo ""
    echo "  [Pass@K Results]"
    echo "  cat checkpoints_v12_qwen_7b_llm_sim/results.json | grep -A 10 'pass_at_k'"
    echo ""
    echo "  [LLM Simulator Statistics]"
    echo "  cat checkpoints_v12_qwen_7b_llm_sim/results.json | grep -A 10 'llm_simulator_stats'"
    echo ""
else
    echo ""
    echo "Training failed! Check error logs:"
    echo "  tail -100 logs/grpo_aco_v12_1_qwen_7b_llm_sim_*.err"
    echo ""
fi

exit $EXIT_CODE