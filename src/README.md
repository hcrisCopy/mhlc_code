# MHLC Pre-Training Data Pipeline

## 0. 先下载原项目发布的 baseline Capability Head 权重

作者发布的 Capability Head 权重都比较小，推荐实验开始先一次性全部下载：

```bash
cd mhlc_code
python src/13_download_baseline_capability_head.py --all
```

默认输出到：

```text
mhlc_data/trained_models/baseline_capability_heads/
```

只想先确认会下载哪些 repo，不真正下载：

```bash
cd mhlc_code
python src/13_download_baseline_capability_head.py --all --dry-run
```

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

当前代码只走纯文本路线，不下载图像数据集。
全部纯文本数据最终约 8GB，运行峰值建议预留 20GB。

下载 Capability Head 的纯文本源数据：

```bash
python src/01_download_data.py --group capability
```

只下载 When2Call：

```bash
python src/01_download_data.py --group when2call
```

只下载可公开下载的纯文本 benchmark 数据：

```bash
python src/01_download_data.py --group benchmarks
```

一次性下载全部纯文本数据：

```bash
python src/01_download_data.py --group all
```

如果你有原 paper/local 的两个 CSV 快照，也可以一并导入到原 benchmark runner
期望的文件名：

```bash
python src/01_download_data.py --group benchmarks \
  --benchmarks math,mmlu_pro \
  --math-csv ../paper_snapshots/merged_math.csv \
  --mmlu-pro-csv ../paper_snapshots/test.csv
```

原仓库没有提供这两个 CSV，也没有提供它们的 preparation scripts。现在默认 `--group all`
或 `--group benchmarks` 会从公开 Hugging Face benchmark 各抽 1000 条生成兼容 CSV：

```text
mhlc_data/data/benchmarks/merged_math.csv
mhlc_data/data/benchmarks/test.csv
```

这两个公开版 CSV 可用于跑 baseline，但不是作者 paper snapshot。需要严格复现作者表格时，仍然优先传入作者原始 CSV。
公开版默认 `--csv-benchmark-sample-size 1000 --csv-benchmark-seed 42`，需要固定别的抽样规模或 seed 时显式传参。
其中 `math` 严格使用 `EleutherAI/hendrycks_math` 的 `test` split，合并 7 个数学类别 config 后抽样生成 `merged_math.csv`；不会退到 train split。
`mmlu_pro` 使用 `TIGER-Lab/MMLU-Pro` 的 `test` split。

## 2. Capability Head: 生成 raw completion 数据

这一步严格复用原仓库 `combined_all_datagen_multimodel.py` 的文本 prompt、
vLLM 参数和保存 schema。源数据只保留原混合 120k 配比里的纯文本部分：
`dapo=10213`、`triviaqa=10213`、`apigen-mt-5k=20425`，合计 `40851`。

```bash
python src/02_generate_capability_raw.py \
  --model-path ../Qwen/Qwen3-VL-4B-Instruct
```

4090 单卡先跑通流程时，可以显式指定每个文本 source 的条数：

```bash
python src/02_generate_capability_raw.py \
  --model-path ../Qwen/Qwen3-VL-4B-Instruct \
  --source-counts dapo=20,triviaqa=20,apigen-mt-5k=20 \
  --gen-chunk-size 16
```

正式全量运行时，去掉 `--source-counts` 和小 batch 参数即可。

默认产物：

```text
mhlc_data/data/train/Qwen3VL/Qwen3_VL_4B_Instruct_text_only_OriginalMixedShare_40851/
  raw/
  selection_manifest.json
  generation_stats.json
```

默认关键参数对齐原项目复现实验：

```text
model_family=qwen3_vl
thinking_mode=off
total_qa_pairs=40851
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
  --run-root ../mhlc_data/data/train/Qwen3VL/Qwen3_VL_4B_Instruct_text_only_smoke_60 \
  --judge-model-id ../Qwen/Qwen3-VL-4B-Instruct \
  --judge-batch-size 8
```

正式复现时只把 `--judge-model-id` 改回上面的 30B judge 路径即可。

默认产物：

```text
mhlc_data/data/train/Qwen3VL/Qwen3_VL_4B_Instruct_text_only_OriginalMixedShare_40851/
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
  --tokenizer-id ../Qwen/Qwen3-VL-4B-Instruct \
  --output-dir ../mhlc_data/data/train/when2call/when2call_processed_4class_smoke \
  --max-rows-per-split 20 \
  --random-sample \
  --batch-size 4 \
  --gpu-memory-utilization 0.70 \
  --max-model-len 16000 \
  --max-tokens 4096
```

正式复现时只把 `--annotator-model-id` 和 `--tokenizer-id` 改回 30B annotator
路径，并去掉 `--max-rows-per-split` 和 smoke 输出目录即可。

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

4090 单卡先跑通流程时：

```bash
python src/05_generate_when2call_completions.py \
  --input-path ../mhlc_data/data/train/when2call/when2call_processed_4class_smoke/when2call_aux_labels.jsonl \
  --output-dir ../mhlc_data/data/train/when2call/qwen3vl/Qwen3-VL-4B-Instruct_4class_smoke \
  --model-path ../Qwen/Qwen3-VL-4B-Instruct \
  --limit 40 \
  --batch-size 8
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

4090 单卡只做 smoke eval completion：

```bash
python src/06_generate_when2call_eval_completions.py \
  --model-path ../Qwen/Qwen3-VL-4B-Instruct \
  --output-path ../mhlc_data/eval_outputs/when2call/Qwen3-VL-4B-Instruct/when2call_test_generated_4class_smoke.parquet \
  --max-eval-rows 20 \
  --batch-size 1 \
  --max-model-len 16000 \
  --max-tokens 2048 \
  --gpu-memory-utilization 0.70 \
  --max-num-seqs 1 \
  --enforce-eager
```

默认输出：

```text
mhlc_data/eval_outputs/when2call/Qwen3-VL-4B-Instruct/when2call_test_generated_4class.parquet
```
