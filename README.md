项目结构目录
环境配置
conda create -n mhlc python=3.10 -y
conda activate mhlc

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
模型与数据准备
下载指令