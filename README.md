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
    Qwen3-VL-30B-A3B-Instruct-FP8
    Qwen3-30B-A3B-Instruct-2507-FP8
环境配置
conda create -n mhlc python=3.10 -y
conda activate mhlc

pip install -U pip
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

说明：Qwen3-VL 需要新版 transformers / vLLM。torch 需要按服务器驱动选择 CUDA wheel；如果 `nvidia-smi` 只支持 CUDA 12.6，把上面的 `cu128` 改成 `cu126`。不要再用旧的 `torch==2.6.0`。
模型与数据准备
下载指令
pip install -U modelscope

mkdir -p ../Qwen/Qwen3-VL-4B-Instruct
mkdir -p ../Qwen/Qwen3-VL-32B-Thinking-FP8

modelscope download --model Qwen/Qwen3-VL-4B-Instruct \
  --local_dir ../Qwen/Qwen3-VL-4B-Instruct

modelscope download --model Qwen/Qwen3-VL-32B-Thinking-FP8 \
  --local_dir ../Qwen/Qwen3-VL-32B-Thinking-FP8

modelscope download \
  --model Qwen/Qwen3-VL-32B-Instruct-FP8 \
  --local_dir ../Qwen/Qwen3-VL-32B-Instruct-FP8
  
mkdir -p ../Qwen/Qwen3-VL-30B-A3B-Instruct-FP8
mkdir -p ../Qwen/Qwen3-30B-A3B-Instruct-2507-FP8

modelscope download --model Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 \
  --local_dir ../Qwen/Qwen3-VL-30B-A3B-Instruct-FP8

modelscope download --model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
  --local_dir ../Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
