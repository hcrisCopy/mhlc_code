章节目录
项目结构目录
mhlc_code
mhlc_data
    data/
        train/
        benchmarks/
    trained_models/
    eval_outputs/
    neurons/
    visualization/
Qwen
    Qwen3-VL-4B-Instruct
    Qwen3-VL-32B-Thinking-FP8
环境配置
conda create -n mhlc python=3.10 -y
conda activate mhlc

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
模型与数据准备
下载指令
pip install -U modelscope

mkdir -p ../Qwen/Qwen3-VL-4B-Instruct
mkdir -p ../Qwen/Qwen3-VL-32B-Thinking-FP8

modelscope download --model Qwen/Qwen3-VL-4B-Instruct \
  --local_dir ../Qwen/Qwen3-VL-4B-Instruct

modelscope download --model Qwen/Qwen3-VL-32B-Thinking-FP8 \
  --local_dir ../Qwen/Qwen3-VL-32B-Thinking-FP8
