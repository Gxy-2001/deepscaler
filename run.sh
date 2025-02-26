
# conda create -n deepscaler1 -y python=3.10
# conda init 
# source deactivate
# conda activate deepscaler1


# python scripts/data/deepscaler_dataset.py

# pip install -e ./verl
# pip install -e .
# pip install wandb
# wandb login


# # ray start --head --node-ip-address 0.0.0.0 --num-gpus 8
# # ray start --head --node-ip-address=$(hostname -I | awk '{print $1}') --port=6379 --num-gpus=8
# ray start --address=10.6.67.173:6379 --num-gpus=8



# cd /scratch/deepscaler
# # 8k context 8 A100-80G
# export VLLM_ATTENTION_BACKEND=XFORMERS
# Run 8K context length training
export MODEL_PATH="/mnt/teamdrive/xy/xy/sft/1205/QWEN15B_iter2345_sft_ALL_filterXwinMath20XwinCom30"
bash scripts/train/run_deepscaler_1.5b_8k.sh --model $MODEL_PATH
# CUDA_LAUNCH_BLOCKING=1 VLLM_ATTENTION_BACKEND=XFORMERS bash scripts/train/run_deepscaler_1.5b_8k_4node.sh --model $MODEL_PATH
python keepgpu.py