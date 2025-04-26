#!/bin/bash
########################################
# SLURM directives
########################################
#SBATCH --job-name=LLaVA_Multi_GPU
#SBATCH --output=llava_multi_gpu_%j.out
#SBATCH --error=llava_multi_gpu_%j.err
#SBATCH --partition=general
#SBATCH --time=2:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A100_80GB:2

# 激活conda环境
source /home/ayliu2/miniconda3/bin/activate llava_2

# 切换到LLaVA-NeXT目录，确保能正确导入
cd /home/ayliu2/LLaVA-NeXT

# 如果需要，安装LLaVA-NeXT
# pip install -e ".[train]"

# 设置变量
MODEL_PATH="/data/user_data/ayliu2/huggingface/transformers/models--lmms-lab--llava-next-qwen-32b/snapshots/406e836b09a80c502c899eb4a429cd86e1d7e9fb"
INPUT_FILE="/data/user_data/ayliu2/WebQA_val.json"
OUTPUT_FILE="/data/user_data/ayliu2/WebQA_results_test.json"
IMAGE_DIR="/data/user_data/ayliu2/WebQA_imgs_7z_chunks/images"
MAX_INSTANCES=20  # 测试20个实例
NUM_GPUS=2        # 使用2个GPU

# 打印信息
echo "开始时间: $(date)"
echo "运行LLaVA-Next多GPU推理测试..."
echo "当前目录: $(pwd)"
echo "Python路径: $PYTHONPATH"
echo "GPU信息:"
nvidia-smi

# 切换到脚本目录
cd /home/ayliu2/LLaVA-NeXT/alex_script

# 运行Python多GPU脚本
python inference_multi_gpu.py \
    --model-path $MODEL_PATH \
    --input-file $INPUT_FILE \
    --output-file $OUTPUT_FILE \
    --image-dir $IMAGE_DIR \
    --max-instances $MAX_INSTANCES \
    --num-gpus $NUM_GPUS

# 打印完成信息
echo "结束时间: $(date)"
echo "输出结果保存在: $OUTPUT_FILE"