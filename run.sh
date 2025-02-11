conda create -n deepscaler -y python=3.10
conda init 
source deactivate
conda activate deepscaler


python scripts/data/deepscaler_dataset.py

pip install -e ./verl
pip install -e .
pip install wandb
wandb login


cd /scratch/deepscaler
# 8k context 8 A100-80G
export VLLM_ATTENTION_BACKEND=XFORMERS
# Run 8K context length training
export MODEL_PATH="/mnt/lyna-selfplay/model/DeepSeek-R1-Distill-Qwen-1.5B"
bash scripts/train/run_deepscaler_1.5b_8k.sh --model $MODEL_PATH