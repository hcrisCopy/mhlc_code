# MHLC 神经元 Head（纯文本复现）

本项目在不修改 `Multi-Head-Latent-Control/` 原项目的前提下，复用其数据、prompt、生成、打标、路由和论文表格指标；在此基础上，以 FFN intermediate 神经元特征训练 Capability / Resolution Head。当前正式主线为 `Qwen3-VL-4B-Instruct`，但只处理纯文本样本。

## 目录与约定

从父目录进入代码目录运行：

```bash
cd mhlc_code
```

```text
mhlc_code/                                      # 本仓库（命令执行位置）
├── README.md                                   # 本文：正式八卡交接入口
├── requirements.txt                            # Python 依赖（torch 另行按 CUDA 安装）
├── src/
│   ├── 01_...py ～ 14_...py                    # 数据、生成、打标、探测、训练、评测流程
│   ├── mhlc_data_prep/                         # 原项目数据与八卡分片/汇总适配
│   ├── mhlc_neuron_probe/                      # 神经元特征、Head 训练、指标与可视化
│   ├── FORMAL_COMMANDS.md                      # 单卡正式命令对照
│   └── SMOKE_COMMANDS.md                       # 小规模冒烟命令
└── Multi-Head-Latent-Control/                  # 原项目只读参考，禁止修改

../mhlc_data/                                   # 运行产物根目录
├── data/
│   ├── sources/{capability,when2call}/          # 物料化的训练/标注源数据
│   ├── benchmarks/                              # TriviaQA、Table2 等评测数据
│   └── train/{Qwen3VL,when2call}/               # completion、标签、verified 数据
├── neurons/<model>/{capability,resolution}/     # selected_neurons、特征分片
├── trained_models/
│   ├── baseline_capability_heads/               # 下载的原作者 Capability Head
│   └── neuron_heads/<model>/{capability,resolution}/  # 本项目训练的 Head
├── eval_outputs/                                # When2Call、Table2/3/4 的逐样本与汇总结果
└── visualization/neurons/<model>/               # 神经元选择图与训练可视化

../Qwen/                                         # 本地 Qwen 模型目录（4B/8B/30B/32B）
```

所有命令和输出均使用相对路径。正式八卡命令使用 `torchrun --nproc_per_node=8`：每个进程仅看到一张卡，完整加载同一个模型，按确定性分片处理 1/8 数据；不会 tensor parallel 拆分模型。rank 0 只负责合并分片、写最终表格和打印与单卡一致的汇总指标。

八卡阶段会保留隐藏的 worker 缓存，通常中断后直接重复同一条命令即可续跑；若中断发生在最终合并时，或要从单卡切换为八卡，在该阶段命令末尾加 `--clean` 重建。不要混用单卡和八卡产物。

## 环境配置

```bash
conda create -n mhlc python=3.10 -y
conda activate mhlc
conda install -y -c conda-forge libstdcxx-ng libgcc-ng sqlite icu
pip install -U pip
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
```

若服务器驱动只支持 CUDA 12.6，将 `cu128` 改为 `cu126`。不要换回旧版 `torch==2.6.0`。

### Flash Attention 配置

Capability / Table2 / TriviaQA 按原 recipe 使用 `flash_attention_3`；Resolution / When2Call 使用 `flash_attention_2`。FlashAttention 扩展需要服务器安装 CUDA Toolkit（含 `nvcc`），不能只有 PyTorch 的 CUDA wheel。先确认环境：

```bash
nvidia-smi
nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
pip install packaging psutil ninja
ninja --version
```

安装 FA2（CUDA 12+，A100/RTX 30/40/H100 等 Ampere、Ada 或 Hopper GPU）：

```bash
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
```

严格使用 FA3 还要求 H100/H800 和 CUDA Toolkit >= 12.3（官方建议 12.8）。在本仓库父目录外准备源码并编译，避免将第三方源码写入本仓库：

```bash
git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git ../flash-attention
cd ../flash-attention/hopper
python setup.py install
cd ../../mhlc_code
```

若编译时因主机内存不足失败，可仅在重新安装时设置较小的编译并发，例如 `MAX_JOBS=4 pip install flash-attn --no-build-isolation`；这不影响训练/评测参数。安装完成后必须检查：

```bash
python -c "from flash_attn_3 import flash_attn_interface; print('fa3 ok')"
python -c "import flash_attn; print('fa2 ok')"
```

非 H100/H800 机器不能严格运行 FA3 recipe。代码支持将 Capability / Table2 / TriviaQA 的 `--attn-implementation` 改为 `sdpa`：它不依赖 FlashAttention，模型、prompt、训练/评测逻辑和指标计算不变，通常可作为兼容性排障路径。代价是 32K 上下文下速度更慢、显存占用更高；能否完成正式全量取决于服务器显存。若出现 OOM，不要擅自缩小正式 batch 或长度，应先记录报错并确认服务器配置。使用 `sdpa` 的结果不再是严格的原作者 attention 配置。

## 模型与数据准备

```bash
pip install -U modelscope

modelscope download --model Qwen/Qwen3-VL-4B-Instruct --local_dir ../Qwen/Qwen3-VL-4B-Instruct
modelscope download --model Qwen/Qwen3-VL-8B-Instruct --local_dir ../Qwen/Qwen3-VL-8B-Instruct
modelscope download --model Qwen/Qwen3-VL-32B-Thinking-FP8 --local_dir ../Qwen/Qwen3-VL-32B-Thinking-FP8
modelscope download --model Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 --local_dir ../Qwen/Qwen3-VL-30B-A3B-Instruct-FP8
modelscope download --model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 --local_dir ../Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
```

模型用途：4B 是训练和评测 backbone；30B-A3B 是 Capability correctness judge；30B-A3B-2507 是 When2Call 四类标签 annotator；32B Thinking 是 Table2 强模型；8B 是 Table2 的正式 judge fallback。

## 正式八卡流程

### 0. 下载原作者 Capability Head

```bash
python src/13_download_baseline_capability_head.py --all
```

输出：`../mhlc_data/trained_models/baseline_capability_heads/`。仅查看下载映射可加 `--dry-run`。

### 1. 下载并物料化数据

```bash
python src/01_download_data.py --group all
```

输出：`../mhlc_data/data/sources/` 和 `../mhlc_data/data/benchmarks/`；其中 Table2 的 `merged_math.csv`、`test.csv` 位于后者根目录。

### 2. 生成 Capability 训练 completion

```bash
torchrun --standalone --nproc_per_node=8 src/02_generate_capability_raw.py \
  --model-path ../Qwen/Qwen3-VL-4B-Instruct \
  --model-family qwen3_vl --thinking-mode off \
  --max-model-len 32768 --gpu-memory-utilization 0.90 \
  --tensor-parallel-size 1 --seed 42 --gen-chunk-size 128 --raw-shard-size 4000
```

输出：`../mhlc_data/data/train/Qwen3VL/Qwen3_VL_4B_Instruct_text_only_OriginalMixedShare_40851/raw/`，以及同目录的 `selection_manifest.json`、`generation_stats.json`。

### 3. Capability correctness 打标

```bash
torchrun --standalone --nproc_per_node=8 src/03_label_capability_raw.py \
  --judge-model-id ../Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 --judge-batch-size 64
```

输出：上一步目录下的 `verified/` 与 `verification_stats.json`。

### 4. 构造 When2Call 四类标签

```bash
torchrun --standalone --nproc_per_node=8 src/04_prepare_when2call_labels.py \
  --annotator-model-id ../Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
  --tokenizer-id ../Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
  --batch-size 256 --max-tokens 16000 --gpu-memory-utilization 0.50 \
  --tensor-parallel-size 1 --max-model-len 32000 --seed 42
```

输出：`../mhlc_data/data/train/when2call/when2call_processed_4class/` 下的 JSONL、Parquet 和统计文件。

### 5. 生成 Resolution 训练 completion

```bash
torchrun --standalone --nproc_per_node=8 src/05_generate_when2call_completions.py \
  --model-path ../Qwen/Qwen3-VL-4B-Instruct --model-family qwen3_vl --thinking-mode off \
  --batch-size 64 --max-tokens 16000 --gpu-memory-utilization 0.90 \
  --tensor-parallel-size 1 --max-model-len 32000 --seed 42
```

输出：`../mhlc_data/data/train/when2call/qwen3vl/Qwen3-VL-4B-Instruct_4class/`。

### 6. 生成 When2Call 测试 completion

```bash
torchrun --standalone --nproc_per_node=8 src/06_generate_when2call_eval_completions.py \
  --model-path ../Qwen/Qwen3-VL-4B-Instruct --model-family qwen3_vl --thinking-mode off \
  --batch-size 64 --max-tokens 16000 --gpu-memory-utilization 0.50 \
  --tensor-parallel-size 1 --max-model-len 32000 --seed 42
```

输出：`../mhlc_data/eval_outputs/when2call/Qwen3-VL-4B-Instruct/when2call_test_generated_4class.parquet`。

### 7. 探测 Capability 神经元

```bash
torchrun --standalone --nproc_per_node=8 src/07_probe_capability_neurons.py \
  --model-path ../Qwen/Qwen3-VL-4B-Instruct --thinking-mode off \
  --attn-implementation flash_attention_3 --batch-size 1 --num-workers 8 \
  --max-seq-len 32000 --top-ratio 0.10
```

输出：`../mhlc_data/neurons/Qwen3-VL-4B-Instruct/capability/` 和 `../mhlc_data/visualization/neurons/Qwen3-VL-4B-Instruct/capability/`。八卡仅合并充分统计量，神经元打分公式和最多 10% 的选择规则不变。

### 8. 训练 Capability 神经元 Head

```bash
torchrun --standalone --nproc_per_node=8 src/08_train_capability_neuron_head.py \
  --model-path ../Qwen/Qwen3-VL-4B-Instruct --thinking-mode off \
  --attn-implementation flash_attention_3 --extract-batch-size 1 --feature-shard-size 512 \
  --train-batch-size 1 --grad-accum-steps 16 --num-epochs 2 --num-workers 8 \
  --lr 1.0e-4 --warmup-ratio 0.03 --min-lr-ratio 0.10 \
  --failure-threshold 0.5 --min-class-weight 0.1 --max-class-weight 10
```

输出：特征缓存 `../mhlc_data/neurons/Qwen3-VL-4B-Instruct/capability/feature_shards/`，模型 `../mhlc_data/trained_models/neuron_heads/Qwen3-VL-4B-Instruct/capability/neuron_head_final.pt`，指标 `final_metrics.json`。八卡并行的是冻结 backbone 特征提取；小型 head 仍由 rank 0 使用原训练循环、原 batch 和原梯度累积参数训练。

### 9. 探测 Resolution 神经元

```bash
torchrun --standalone --nproc_per_node=8 src/09_probe_resolution_neurons.py \
  --model-path ../Qwen/Qwen3-VL-4B-Instruct --thinking-mode off \
  --attn-implementation flash_attention_2 --batch-size 1 --num-workers 4 \
  --max-seq-len 32000 --top-ratio 0.10
```

输出：`../mhlc_data/neurons/Qwen3-VL-4B-Instruct/resolution/` 和对应 `visualization/` 目录。

### 10. 训练 Resolution 神经元 Head

```bash
torchrun --standalone --nproc_per_node=8 src/10_train_resolution_neuron_head.py \
  --model-path ../Qwen/Qwen3-VL-4B-Instruct --thinking-mode off \
  --attn-implementation flash_attention_2 --extract-batch-size 1 --feature-shard-size 512 \
  --train-batch-size 1 --grad-accum-steps 16 --num-epochs 5 --num-workers 4 \
  --lr 1.0e-4 --warmup-ratio 0.03 --min-lr-ratio 0.10 \
  --decision-threshold 0.5 --min-class-weight 1.0 --max-class-weight 10
```

输出：`../mhlc_data/trained_models/neuron_heads/Qwen3-VL-4B-Instruct/resolution/neuron_head_final.pt` 与 `final_metrics.json`。

### 11. TriviaQA / Table4

```bash
torchrun --standalone --nproc_per_node=8 src/11_eval_capability_triviaqa_neuron_head.py \
  --model-path ../Qwen/Qwen3-VL-4B-Instruct --thinking-mode off \
  --attn-implementation flash_attention_3 --thresholds 0.5,0.6,0.7,0.8,0.9,0.95 \
  --generation-batch-size 64 --score-batch-size 1 \
  --vllm-max-model-len 32768 --vllm-gpu-memory-utilization 0.40 --vllm-max-num-seqs 128
```

输出：`../mhlc_data/eval_outputs/neuron_heads/Qwen3-VL-4B-Instruct/capability_triviaqa/`。终端打印原 Table4 风格的 `No-Tool`、`Score`、`Calls`、`Precision`、`Missed`。

### 12. When2Call / Table3

```bash
torchrun --standalone --nproc_per_node=8 src/12_eval_resolution_when2call_neuron_head.py \
  --model-path ../Qwen/Qwen3-VL-4B-Instruct --thinking-mode off \
  --attn-implementation flash_attention_2 --batch-size 1 --decision-threshold 0.1
```

输出：`../mhlc_data/eval_outputs/neuron_heads/Qwen3-VL-4B-Instruct/resolution_when2call/`。终端打印原 Table3 风格的 `F1`、`Acc` 和三行比较。

### 13. Table2 三项纯文本 benchmark

```bash
torchrun --standalone --nproc_per_node=8 src/14_eval_capability_table2_textbench.py \
  --model1-path ../Qwen/Qwen3-VL-4B-Instruct \
  --model2-path ../Qwen/Qwen3-VL-32B-Thinking-FP8 \
  --model1-thinking-mode off --model2-thinking-mode on \
  --baseline-head-path ../mhlc_data/trained_models/baseline_capability_heads/Qwen__Qwen3-VL-4B-Instruct/full/capability_head.pt \
  --attn-implementation flash_attention_3 --thresholds 0.5,0.6,0.7,0.8,0.9 \
  --vllm-max-model-len 32768 \
  --model1-generation-batch-size 128 --model2-generation-batch-size 64 \
  --model1-max-num-seqs 128 --model2-max-num-seqs 64 \
  --model1-gpu-memory-utilization 0.70 --model2-gpu-memory-utilization 0.80 \
  --judge-model-path ../Qwen/Qwen3-VL-8B-Instruct --judge-batch-size 32 \
  --m2-input-cost-per-1m-usd 0.70 --m2-output-cost-per-1m-usd 8.40
```

输出：`../mhlc_data/eval_outputs/neuron_heads/Qwen3-VL-4B-Instruct/capability_table2_textbench/`。其中 `table2_textbench_comparison*.csv` 是最终表，`triviaqa/`、`math/`、`mmlu_pro/` 保存逐样本生成、head 分数、路由和 judge 结果。`cost` 只按路由到 model2 的 token 和给定单价计算，不会产生 API 费用。

## 结果与实现说明

Capability 使用已打好的 `correctness_score`；Resolution 保持原项目四类行为和 3-sigmoid 目标：`tool_call`、`request_for_info`、`cannot_answer`、`direct_answer`。神经元定义为各层 MLP `down_proj` 输入的 completion-token 平均激活；只选取得分最高且总量不超过 FFN intermediate 10% 的坐标。

Capability 最终输出 `roc_auc`、`aupr_c`、`aupr_i`、`ece`；Resolution 输出 `macro_f1`、`accuracy` 与混淆矩阵。图表均为本地 PNG，不使用 wandb。

快速小样本排障命令见 `src/SMOKE_COMMANDS.md`；完整单卡参数对照见 `src/FORMAL_COMMANDS.md`。本 README 的八卡命令才是正式提交命令。
