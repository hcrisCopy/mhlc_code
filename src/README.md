# MHLC Pre-Training Data Pipeline

这些脚本只处理“训练前”的数据阶段，不训练 head，也不改动
`Multi-Head-Latent-Control/` 原仓库。

推荐在远程服务器上从 `mhlc_code` 目录运行：

```bash
cd mhlc_code
```

## 目录约定

默认相对路径：

```text
mhlc_code/
  Multi-Head-Latent-Control/
  src/
mhlc_data/
  data/
    sources/
    train/
    benchmarks/
  trained_models/
  eval_outputs/
Qwen/
  Qwen3-VL-4B-Instruct/
  Qwen3-VL-32B-Thinking-FP8/
```

HF 的临时 runtime cache 会被放在：

```text
mhlc_data/downloads/hf_runtime_cache/
```

`01_download_data.py` 默认在成功结束后删除这个临时目录。正式数据会物料化到
`mhlc_data/data/sources/...` 或 `mhlc_data/data/benchmarks/...`。

## 续跑和清理

默认命令按正式参数运行，并尽量复用已有产物：

- 下载阶段已完成的数据会跳过；
- Capability raw 会接着已有 raw shard 续跑；
- Capability labeling 会跳过已有 verified shard；
- When2Call labeling 和 completion 默认使用原项目的 `--resume`；
- 需要重跑某阶段时，在对应命令后加 `--clean`。

## 1. 下载 / 物料化原始数据

```bash
python src/01_download_data.py --group all
```

只下载 Capability Head 训练前所需源数据：

```bash
python src/01_download_data.py --group capability
```

只下载 When2Call：

```bash
python src/01_download_data.py --group when2call
```

只下载 Table 2 benchmark 数据：

```bash
python src/01_download_data.py --group benchmarks
```

如果你有原 paper/local 的两个 CSV 快照，也可以一并导入到原 benchmark runner
期望的文件名：

```bash
python src/01_download_data.py --group benchmarks \
  --benchmarks math,mmlu_pro \
  --math-csv ../paper_snapshots/merged_math.csv \
  --mmlu-pro-csv ../paper_snapshots/test.csv
```

原仓库没有提供这两个 CSV，也没有提供它们的 preparation scripts；没有 paper
snapshot 时不用运行这条命令。默认 `--group benchmarks` 只物料化可下载的
HF-backed benchmark 数据。

## 2. Capability Head: 生成 raw completion 数据

这一步严格复用原仓库 `combined_all_datagen_multimodel.py` 的采样比例、prompt、
vLLM 参数和保存 schema，只把数据读取改成 `mhlc_data/data/sources/capability/...`。

```bash
python src/02_generate_capability_raw.py \
  --model-path ../Qwen/Qwen3-VL-4B-Instruct
```

默认产物：

```text
mhlc_data/data/train/Qwen3VL/Qwen3_VL_4B_Instruct_hard_Mixed_Sources_120k/
  raw/
  selection_manifest.json
  generation_stats.json
```

默认关键参数对齐原项目复现实验：

```text
model_family=qwen3_vl
thinking_mode=off
total_qa_pairs=120000
max_model_len=32768
gpu_memory_utilization=0.90
tensor_parallel_size=1
seed=42
gen_chunk_size=128
raw_shard_size=4000
```

## 3. Capability Head: 给 raw completion 打 correctness_score

这一步严格复用原仓库 `combined_all_labeling_multimodel.py` 的规则和 judge prompt。

```bash
python src/03_label_capability_raw.py
```

默认 judge 模型和原仓库一致：

```text
Qwen/Qwen3-VL-30B-A3B-Instruct-FP8
```

如果你也把 judge 模型下载到了本地，可以改成：

```bash
python src/03_label_capability_raw.py \
  --judge-model-id ../Qwen/Qwen3-VL-30B-A3B-Instruct-FP8
```

24G 4090 只想先跑通流程时，可以临时用本地小模型当 judge。除了模型路径，
label 规则、judge prompt、batch 参数和原项目保持一致：

```bash
python src/03_label_capability_raw.py \
  --judge-model-id ../Qwen/Qwen3-VL-4B-Instruct
```

正式复现时只把 `--judge-model-id` 改回上面的 30B judge 路径即可。

默认产物：

```text
mhlc_data/data/train/Qwen3VL/Qwen3_VL_4B_Instruct_hard_Mixed_Sources_120k/
  verified/
  verification_stats.json
```

后续 Capability Head 训练读的就是 `verified/`，标签列是 `correctness_score`。

## 4. When2Call / Resolution Head: 构造 4 类标签

这一步严格复用原仓库 `when2call/when2call_build_head_labels_4class.py` 的提示词、
类别定义和 annotator 设置。

```bash
python src/04_prepare_when2call_labels.py
```

默认 annotator 和原仓库一致：

```text
Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
```

如果你也把 annotator 下载到了本地，正式复现可以显式传本地路径：

```bash
python src/04_prepare_when2call_labels.py \
  --annotator-model-id ../Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
  --tokenizer-id ../Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
```

24G 4090 只想先跑通流程时，可以临时把 annotator 换成本地小模型。除了模型路径，
提示词、类别定义、采样参数和原项目保持一致：

```bash
python src/04_prepare_when2call_labels.py \
  --annotator-model-id ../Qwen/Qwen3-VL-4B-Instruct \
  --tokenizer-id ../Qwen/Qwen3-VL-4B-Instruct
```

正式复现时只把 `--annotator-model-id` 和 `--tokenizer-id` 改回 30B annotator
路径即可。

默认产物：

```text
mhlc_data/data/train/when2call/when2call_processed_4class/
  when2call_aux_labels.jsonl
  when2call_aux_labels.parquet
  when2call_aux_labels_stats.json
```

## 5. When2Call / Resolution Head: 用 4B 生成 completion

这一步严格复用原仓库 `when2call/when2call_generate_completions_4class.py`，
模型改成你的本地 `Qwen3-VL-4B-Instruct`。

```bash
python src/05_generate_when2call_completions.py \
  --model-path ../Qwen/Qwen3-VL-4B-Instruct
```

默认产物：

```text
mhlc_data/data/train/when2call/qwen3vl/Qwen3-VL-4B-Instruct_4class/
```

后续 Table 3 / Resolution Head 训练读的就是这个目录。

## 可选：生成 When2Call test completion

这是评测阶段会用的，不属于训练前必需数据：

```bash
python src/06_generate_when2call_eval_completions.py \
  --model-path ../Qwen/Qwen3-VL-4B-Instruct
```

默认输出：

```text
mhlc_data/eval_outputs/when2call/Qwen3-VL-4B-Instruct/when2call_test_generated_4class.parquet
```
