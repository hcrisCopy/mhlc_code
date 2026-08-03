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
conda install -y -c conda-forge libstdcxx-ng libgcc-ng sqlite icu
pip install -U pip
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

说明：Qwen3-VL 需要新版 transformers / vLLM。torch 需要按服务器驱动选择 CUDA wheel；如果 `nvidia-smi` 只支持 CUDA 12.6，把上面的 `cu128` 改成 `cu126`。不要再用旧的 `torch==2.6.0`。
Flash Attention 说明：正式命令为了对齐原作者 recipe，Capability / Table2 / TriviaQA 使用 `flash_attention_3`，Resolution / When2Call 使用 `flash_attention_2`。这两个不是 `requirements.txt` 里强制安装的基础依赖。正式 H20 服务器运行前先检查：
python -c "from flash_attn_3 import flash_attn_interface; print('fa3 ok')"
python -c "import flash_attn; print('fa2 ok')"
如果缺 FA2 / FA3，先确认编译环境：
python -c "import torch; print(torch.__version__, torch.version.cuda)"
nvcc -V
如果缺 FA2：
pip install -U wheel setuptools ninja packaging psutil
MAX_JOBS=8 pip install flash-attn --no-build-isolation
python -c "import flash_attn; print('fa2 ok')"
如果缺 FA3：
git clone https://github.com/Dao-AILab/flash-attention.git flash-attention-src
cd flash-attention-src/hopper
pip install -U wheel setuptools ninja packaging psutil
MAX_JOBS=8 python setup.py install
python -c "from flash_attn_3 import flash_attn_interface; print('fa3 ok')"
FA2 / FA3 需要匹配的 CUDA / PyTorch 编译环境，FA3 还需要 Hopper 系显卡，安装时间可能较长。如果 `nvcc` 不存在或 H20 上暂时装不上，就把正式命令里的 `--attn-implementation` 临时改成 `sdpa` 作为兼容兜底；后者不是严格原作者 attention 参数。
模型与数据准备
下载指令
pip install -U modelscope

modelscope download --model Qwen/Qwen3-VL-4B-Instruct \
  --local_dir ../Qwen/Qwen3-VL-4B-Instruct

modelscope download \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --local_dir ../Qwen/Qwen3-VL-8B-Instruct

modelscope download --model Qwen/Qwen3-VL-32B-Thinking-FP8 \
  --local_dir ../Qwen/Qwen3-VL-32B-Thinking-FP8

modelscope download \
  --model Qwen/Qwen3-VL-32B-Instruct-FP8 \
  --local_dir ../Qwen/Qwen3-VL-32B-Instruct-FP8
  

modelscope download --model Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 \
  --local_dir ../Qwen/Qwen3-VL-30B-A3B-Instruct-FP8

modelscope download --model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
  --local_dir ../Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
正式运行的八卡指令
