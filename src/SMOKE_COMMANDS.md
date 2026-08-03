# 跑通命令

这些命令用于 4090 24G 单卡先跑通全流程，不代表正式实验指标。正式实验看 `src/FORMAL_COMMANDS.md`。

从项目根目录运行：

```bash
cd mhlc_code
```

默认使用：

```text
../Qwen/Qwen3-VL-4B-Instruct
```

## 0. 下载原作者 Capability Head

Table2 smoke 需要原作者 baseline head 做对比，所以先下载权重。

```bash
python src/13_download_baseline_capability_head.py --all
```

输出到：

```text
../mhlc_data/trained_models/baseline_capability_heads/
```

只想先看会下载哪些 repo，不真正下载：

```bash
python src/13_download_baseline_capability_head.py --all --dry-run
```

## 1. 下载数据

跑通全流程也需要 Capability、When2Call 和 benchmark 数据。

```bash
python src/01_download_data.py --group all
```

输出到：

```text
../mhlc_data/data/sources/
../mhlc_data/data/benchmarks/
```

如果只想先下载某一块，可以分别跑：

```bash
python src/01_download_data.py --group capability
python src/01_download_data.py --group when2call
python src/01_download_data.py --group benchmarks
```

如果有论文或作者快照版的 `math`、`mmlu_pro` CSV，导入到 benchmark 目录：

```bash
python src/01_download_data.py --group benchmarks \
  --benchmarks math,mmlu_pro \
  --math-csv ../paper_snapshots/merged_math.csv \
  --mmlu-pro-csv ../paper_snapshots/test.csv
```

输出到：

```text
../mhlc_data/data/benchmarks/merged_math.csv
../mhlc_data/data/benchmarks/test.csv
```

## 2. 生成 Capability 小样本 raw completion

这一步每个文本源只取 20 条，生成 60 条 raw completion。

```bash
python src/02_generate_capability_raw.py \
  --model-path ../Qwen/Qwen3-VL-4B-Instruct \
  --source-counts dapo=20,triviaqa=20,apigen-mt-5k=20 \
  --gen-chunk-size 16
```

输出到：

```text
../mhlc_data/data/train/Qwen3VL/Qwen3_VL_4B_Instruct_text_only_smoke_60/raw/
../mhlc_data/data/train/Qwen3VL/Qwen3_VL_4B_Instruct_text_only_smoke_60/selection_manifest.json
../mhlc_data/data/train/Qwen3VL/Qwen3_VL_4B_Instruct_text_only_smoke_60/generation_stats.json
```

## 3. 给 Capability 小样本打分

这一步临时用 4B 当 judge，只为了跑通流程。

```bash
python src/03_label_capability_raw.py \
  --run-root ../mhlc_data/data/train/Qwen3VL/Qwen3_VL_4B_Instruct_text_only_smoke_60 \
  --judge-model-id ../Qwen/Qwen3-VL-4B-Instruct \
  --judge-batch-size 8
```

输出到：

```text
../mhlc_data/data/train/Qwen3VL/Qwen3_VL_4B_Instruct_text_only_smoke_60/verified/
../mhlc_data/data/train/Qwen3VL/Qwen3_VL_4B_Instruct_text_only_smoke_60/verification_stats.json
```

## 4. 构造 Resolution 小样本标签

这一步临时用 4B 当 annotator，只抽少量 When2Call 数据。

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

输出到：

```text
../mhlc_data/data/train/when2call/when2call_processed_4class_smoke/when2call_aux_labels.jsonl
../mhlc_data/data/train/when2call/when2call_processed_4class_smoke/when2call_aux_labels.parquet
../mhlc_data/data/train/when2call/when2call_processed_4class_smoke/when2call_aux_labels_stats.json
```

## 5. 生成 Resolution 小样本训练 completion

这一步给 smoke 版 When2Call 标签生成 backbone completion。

```bash
python src/05_generate_when2call_completions.py \
  --input-path ../mhlc_data/data/train/when2call/when2call_processed_4class_smoke/when2call_aux_labels.jsonl \
  --output-dir ../mhlc_data/data/train/when2call/qwen3vl/Qwen3-VL-4B-Instruct_4class_smoke \
  --model-path ../Qwen/Qwen3-VL-4B-Instruct \
  --limit 40 \
  --batch-size 8
```

输出到：

```text
../mhlc_data/data/train/when2call/qwen3vl/Qwen3-VL-4B-Instruct_4class_smoke/
```

## 6. 探测 Capability 神经元

这一步直接读取前面生成的 smoke `verified/` 数据。

```bash
python src/07_probe_capability_neurons.py \
  --model-path ../Qwen/Qwen3-VL-4B-Instruct \
  --dataset-path ../mhlc_data/data/train/Qwen3VL/Qwen3_VL_4B_Instruct_text_only_smoke_60/verified \
  --attn-implementation flash_attention_3 \
  --max-seq-len 16000 \
  --batch-size 1 \
  --clean
```

输出到：

```text
../mhlc_data/neurons/Qwen3-VL-4B-Instruct/capability/
../mhlc_data/visualization/neurons/Qwen3-VL-4B-Instruct/capability/
```

## 7. 训练 Capability 神经元 Head

这一步用 smoke 神经元特征训练 Capability Head。

```bash
python src/08_train_capability_neuron_head.py \
  --model-path ../Qwen/Qwen3-VL-4B-Instruct \
  --dataset-path ../mhlc_data/data/train/Qwen3VL/Qwen3_VL_4B_Instruct_text_only_smoke_60/verified \
  --attn-implementation sdpa \
  --max-seq-len 16000 \
  --extract-batch-size 1 \
  --train-batch-size 16
```

输出到：

```text
../mhlc_data/neurons/Qwen3-VL-4B-Instruct/capability/feature_shards/
../mhlc_data/trained_models/neuron_heads/Qwen3-VL-4B-Instruct/capability/neuron_head_final.pt
../mhlc_data/trained_models/neuron_heads/Qwen3-VL-4B-Instruct/capability/final_metrics.json
```

## 8. 探测 Resolution 神经元

这一步直接读取前面生成的 smoke When2Call completion。

```bash
python src/09_probe_resolution_neurons.py \
  --model-path ../Qwen/Qwen3-VL-4B-Instruct \
  --dataset-path ../mhlc_data/data/train/when2call/qwen3vl/Qwen3-VL-4B-Instruct_4class_smoke \
  --attn-implementation sdpa \
  --max-seq-len 16000 \
  --batch-size 1
```

输出到：

```text
../mhlc_data/neurons/Qwen3-VL-4B-Instruct/resolution/
../mhlc_data/visualization/neurons/Qwen3-VL-4B-Instruct/resolution/
```

## 9. 训练 Resolution 神经元 Head

这一步用 smoke 神经元特征训练 Resolution Head。

```bash
python src/10_train_resolution_neuron_head.py \
  --model-path ../Qwen/Qwen3-VL-4B-Instruct \
  --dataset-path ../mhlc_data/data/train/when2call/qwen3vl/Qwen3-VL-4B-Instruct_4class_smoke \
  --attn-implementation sdpa \
  --max-seq-len 16000 \
  --extract-batch-size 1 \
  --train-batch-size 16
```

输出到：

```text
../mhlc_data/neurons/Qwen3-VL-4B-Instruct/resolution/feature_shards/
../mhlc_data/trained_models/neuron_heads/Qwen3-VL-4B-Instruct/resolution/neuron_head_final.pt
../mhlc_data/trained_models/neuron_heads/Qwen3-VL-4B-Instruct/resolution/final_metrics.json
```

## 10. 生成 When2Call 测试小样本 completion

这一步准备 Resolution smoke 评测用的测试 completion。

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

输出到：

```text
../mhlc_data/eval_outputs/when2call/Qwen3-VL-4B-Instruct/when2call_test_generated_4class_smoke.parquet
```

## 11. 评测 Capability Head：TriviaQA / Table4 smoke

这一步只取 20 条 TriviaQA，终端打印 Table4 风格的三行对比。

```bash
python src/11_eval_capability_triviaqa_neuron_head.py \
  --model-path ../Qwen/Qwen3-VL-4B-Instruct \
  --attn-implementation sdpa \
  --max-samples 20 \
  --generation-batch-size 4 \
  --score-batch-size 1 \
  --max-seq-len 16000 \
  --vllm-max-model-len 16000 \
  --vllm-gpu-memory-utilization 0.70 \
  --vllm-max-num-seqs 8 \
  --output-dir ../mhlc_data/eval_outputs/neuron_heads/Qwen3-VL-4B-Instruct/capability_triviaqa_smoke
```

输出到：

```text
../mhlc_data/eval_outputs/neuron_heads/Qwen3-VL-4B-Instruct/capability_triviaqa_smoke/
```

## 12. 评测 Resolution Head：When2Call / Table3 smoke

这一步读取 smoke 测试 completion，终端打印 Table3 风格的三行对比。

```bash
python src/12_eval_resolution_when2call_neuron_head.py \
  --model-path ../Qwen/Qwen3-VL-4B-Instruct \
  --generated-eval-path ../mhlc_data/eval_outputs/when2call/Qwen3-VL-4B-Instruct/when2call_test_generated_4class_smoke.parquet \
  --attn-implementation sdpa \
  --decision-threshold 0.1 \
  --max-seq-len 16000 \
  --batch-size 1 \
  --output-dir ../mhlc_data/eval_outputs/neuron_heads/Qwen3-VL-4B-Instruct/resolution_when2call_smoke
```

输出到：

```text
../mhlc_data/eval_outputs/neuron_heads/Qwen3-VL-4B-Instruct/resolution_when2call_smoke/
```

## 13. 评测 Capability Head：三项纯文本 benchmark / Table2 smoke

这一步每个 benchmark 只取 2 条，并把 `model2` 也临时设成 4B，只为了确认流程能跑完。

```bash
python src/14_eval_capability_table2_textbench.py \
  --model1-path ../Qwen/Qwen3-VL-4B-Instruct \
  --model2-path ../Qwen/Qwen3-VL-4B-Instruct \
  --model2-thinking-mode off \
  --baseline-head-path ../mhlc_data/trained_models/baseline_capability_heads/Qwen__Qwen3-VL-4B-Instruct/full/capability_head.pt \
  --attn-implementation sdpa \
  --max-samples 2 \
  --vllm-max-model-len 4096 \
  --max-seq-len 4096 \
  --generation-max-new-tokens 512 \
  --model1-generation-batch-size 1 \
  --model2-generation-batch-size 1 \
  --model1-max-num-seqs 1 \
  --model2-max-num-seqs 1 \
  --score-batch-size 1 \
  --model1-gpu-memory-utilization 0.70 \
  --model2-gpu-memory-utilization 0.70 \
  --output-dir ../mhlc_data/eval_outputs/neuron_heads/Qwen3-VL-4B-Instruct/capability_table2_textbench_smoke
```

输出到：

```text
../mhlc_data/eval_outputs/neuron_heads/Qwen3-VL-4B-Instruct/capability_table2_textbench_smoke/
```

只想更快确认单个数据集时，在命令后加：

```bash
--benchmarks triviaqa
```

如果 vLLM CUDA graph 初始化阶段仍然显存不够，再加：

```bash
--vllm-enforce-eager
```

如果不是读 smoke 目录，而是读全量目录抽样确认，也可以在神经元探测和训练命令后加：

```bash
--max-samples 64
```

## 续跑和重跑

默认会复用已有产物。想重跑某一步，在对应命令最后加：

```bash
--clean
```

Table2 想强制重新生成中间产物，可以加：

```bash
--no-reuse-generations
--no-reuse-scores
--no-reuse-evaluations
```

所有可视化都保存为本地图片，不使用 wandb。
