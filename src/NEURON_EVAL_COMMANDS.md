# 神经元 Head 评测命令

从项目根目录运行：

```bash
cd mhlc_code
```

默认模型：

```text
../Qwen/Qwen3-VL-4B-Instruct
```

评测脚本默认读取前面训练阶段的产物：

```text
../mhlc_data/neurons/Qwen3-VL-4B-Instruct/capability/selected_neurons.jsonl
../mhlc_data/trained_models/neuron_heads/Qwen3-VL-4B-Instruct/capability/neuron_head_final.pt
../mhlc_data/neurons/Qwen3-VL-4B-Instruct/resolution/selected_neurons.jsonl
../mhlc_data/trained_models/neuron_heads/Qwen3-VL-4B-Instruct/resolution/neuron_head_final.pt
```

## 1. Capability Head：TriviaQA / Table 4

原论文没有提供 `math` 和 `mmlu_pro` 的公开评测快照，这里只跑 `triviaqa`，并按原项目
`multi_agenT_bench/triviaqa_web_overuse_eval.py` 的公式计算：

- `No-Tool`：标准无工具回答的正确率。
- `Score`：按 head 决策是否调用工具后的模拟正确率；调用工具时按原项目设定视为可由工具得到正确答案。
- `Calls`：head 触发的工具调用次数。
- `Precision (%)`：触发调用中，原本无工具会答错、因此确实需要工具的比例。
- `Missed`：原本无工具会答错，但 head 没有触发工具的样本数。

如果还没有物料化 TriviaQA benchmark，先运行：

```bash
python src/01_download_data.py --group benchmarks --benchmarks triviaqa
```

正式评测：

```bash
python src/11_eval_capability_triviaqa_neuron_head.py \
  --model-path ../Qwen/Qwen3-VL-4B-Instruct \
  --attn-implementation sdpa
```

4090 单卡 smoke 评测：

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

默认输出：

```text
../mhlc_data/eval_outputs/neuron_heads/Qwen3-VL-4B-Instruct/capability_triviaqa/
```

关键结果：

```text
summary.json
paper_table4_comparison.json
paper_table4_comparison.csv
threshold_summary.csv
merged_per_example_rows.jsonl
plots/threshold_score_calls.png
```

终端会打印三行对比：

```text
Backbone Choice
Backbone + Capability Head（MHLC）
Backbone + Capability Head（Ours）
```

其中前两行直接使用论文 Table 4 的数值，`Ours` 行由当前运行结果计算。

## 2. Resolution Head：When2Call / Table 3

Resolution 评测读取 `src/06_generate_when2call_eval_completions.py` 的 completion 产物，并按原项目
`when2call/eval/eval_when2call_head_only_4class_3sigmoid.py` 的口径计算：

- head 输出 3 个 sigmoid：`tool_call`、`request_for_info`、`cannot_answer`。
- 如果 3 个分数的最大值低于 `decision_threshold`，预测为第 4 类 `direct_answer`。
- 指标使用 4 类混淆矩阵上的 `macro_f1` 和 `accuracy`。

如果还没有正式 eval completion，先运行：

```bash
python src/06_generate_when2call_eval_completions.py \
  --model-path ../Qwen/Qwen3-VL-4B-Instruct
```

正式评测：

```bash
python src/12_eval_resolution_when2call_neuron_head.py \
  --model-path ../Qwen/Qwen3-VL-4B-Instruct \
  --attn-implementation sdpa \
  --decision-threshold 0.1
```

这里显式传 `--decision-threshold 0.1` 是为了对齐原项目 Table 3 评测脚本默认值。若不传该参数，脚本会优先读取当前 checkpoint 中保存的阈值。

4090 单卡 smoke eval completion：

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

4090 单卡 smoke 评测：

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

默认输出：

```text
../mhlc_data/eval_outputs/neuron_heads/Qwen3-VL-4B-Instruct/resolution_when2call/
```

关键结果：

```text
summary.json
metrics.json
predictions.jsonl
probability_summary.json
paper_table3_comparison.json
paper_table3_comparison.csv
plots/head_confusion.png
```

终端会打印三行对比：

```text
Backbone Choice
Backbone + Resolution Head（MHLC）
Backbone + Resolution Head（Ours）
```

前两行直接使用论文 Table 3 的数值，`Ours` 行由当前运行结果计算。

## 3. 续跑和清理

Capability 评测默认复用已经生成和已经打分的中间结果：

```text
standard_no_tool_rows_raw_generation.jsonl
tool_enabled_rows_raw_generation.jsonl
standard_no_tool_rows_scored_with_neuron_head.jsonl
```

Resolution 评测默认复用已经完成的 head 分数：

```text
predictions.jsonl
```

如果要重跑某个输出目录，命令后追加：

```bash
--clean
```

所有可视化都保存为本地图片，不使用 wandb。
