# 正式实验命令

从项目根目录运行：

```bash
cd mhlc_code
```

默认使用：

```text
../Qwen/Qwen3-VL-4B-Instruct
```

本地模型用途：

```text
../Qwen/Qwen3-VL-4B-Instruct：当前 backbone，生成训练/评测 completion，并训练/评测我们的 4B 神经元 head。
../Qwen/Qwen3-VL-8B-Instruct：Table2 文本 benchmark 的 judge fallback，按原项目评测口径使用。
../Qwen/Qwen3-VL-32B-Thinking-FP8：Table2 的强模型 model2，thinking mode 保持 on。
../Qwen/Qwen3-VL-32B-Instruct-FP8：32B Instruct 相关 baseline/扩展实验备用，不是当前 4B 主线默认模型。
../Qwen/Qwen3-VL-30B-A3B-Instruct-FP8：Capability raw correctness_score judge。
../Qwen/Qwen3-30B-A3B-Instruct-2507-FP8：When2Call 四类标签 annotator，纯文本模型。
```

正式换模型时，主要改各命令里的 `--model-path`、`--model1-path`、`--model2-path`。默认命令会复用已有产物；要重跑某一步，在对应命令最后加 `--clean`。

Attention 后端按原作者 recipe 对齐：Capability / Table2 / TriviaQA 使用 `flash_attention_3`，Resolution / When2Call 使用 `flash_attention_2`。如果服务器加载模型时明确报 flash attention 不支持，再把对应命令临时改成 `sdpa` 作为兼容兜底；这不是严格原作者参数。

## 0. 下载原作者 Capability Head

这一步下载原作者发布的 baseline Capability Head，后面 Table2 会用它和我们的神经元 head 对比。

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

## 1. 下载和物料化数据

一次性准备 Capability、When2Call 和 benchmark 纯文本数据。`math`、`mmlu_pro` 会按下载脚本固定种子抽样并物料化成对应 CSV。

```bash
python src/01_download_data.py --group all
```

输出到：

```text
../mhlc_data/data/sources/
../mhlc_data/data/benchmarks/
```

输出到：

```text
../mhlc_data/data/benchmarks/merged_math.csv
../mhlc_data/data/benchmarks/test.csv
```

## 2. 生成 Capability 训练 raw completion

这一步复用原项目文本 prompt、vLLM 参数和保存格式，用 4B backbone 给训练问题生成回答。

```bash
python src/02_generate_capability_raw.py \
  --model-path ../Qwen/Qwen3-VL-4B-Instruct
```

输出到：

```text
../mhlc_data/data/train/Qwen3VL/Qwen3_VL_4B_Instruct_text_only_OriginalMixedShare_40851/raw/
../mhlc_data/data/train/Qwen3VL/Qwen3_VL_4B_Instruct_text_only_OriginalMixedShare_40851/selection_manifest.json
../mhlc_data/data/train/Qwen3VL/Qwen3_VL_4B_Instruct_text_only_OriginalMixedShare_40851/generation_stats.json
```

## 3. 给 Capability raw 数据打分

这一步复用原项目 judge prompt 和规则，给每条回答打 `correctness_score`。

```bash
python src/03_label_capability_raw.py \
  --judge-model-id ../Qwen/Qwen3-VL-30B-A3B-Instruct-FP8
```

输出到：

```text
../mhlc_data/data/train/Qwen3VL/Qwen3_VL_4B_Instruct_text_only_OriginalMixedShare_40851/verified/
../mhlc_data/data/train/Qwen3VL/Qwen3_VL_4B_Instruct_text_only_OriginalMixedShare_40851/verification_stats.json
```

后续 Capability 神经元探测和训练读取这个 `verified/` 目录。

## 4. 构造 Resolution 四类标签

这一步复用原项目 When2Call 的类别定义、prompt 和 annotator 设置，得到四类监督标签。

```bash
python src/04_prepare_when2call_labels.py \
  --annotator-model-id ../Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
  --tokenizer-id ../Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
```

输出到：

```text
../mhlc_data/data/train/when2call/when2call_processed_4class/when2call_aux_labels.jsonl
../mhlc_data/data/train/when2call/when2call_processed_4class/when2call_aux_labels.parquet
../mhlc_data/data/train/when2call/when2call_processed_4class/when2call_aux_labels_stats.json
```

## 5. 生成 Resolution 训练 completion

这一步用 backbone 给 When2Call 训练集生成 completion，后面 Resolution 神经元探测和训练会读这个目录。

```bash
python src/05_generate_when2call_completions.py \
  --model-path ../Qwen/Qwen3-VL-4B-Instruct
```

输出到：

```text
../mhlc_data/data/train/when2call/qwen3vl/Qwen3-VL-4B-Instruct_4class/
```

## 6. 探测 Capability 神经元

这一步全层扫描神经元，用 `correctness_score` 找和“会不会答对”最相关的神经元，并保存热力图。

```bash
python src/07_probe_capability_neurons.py \
  --model-path ../Qwen/Qwen3-VL-4B-Instruct \
  --thinking-mode off \
  --attn-implementation flash_attention_3 \
  --num-workers 8
```

输出到：

```text
../mhlc_data/neurons/Qwen3-VL-4B-Instruct/capability/
../mhlc_data/visualization/neurons/Qwen3-VL-4B-Instruct/capability/
```

## 7. 训练 Capability 神经元 Head

这一步只取探测出的神经元特征训练 head，不再使用整层 hidden states。

```bash
python src/08_train_capability_neuron_head.py \
  --model-path ../Qwen/Qwen3-VL-4B-Instruct \
  --thinking-mode off \
  --attn-implementation flash_attention_3 \
  --train-batch-size 1 \
  --grad-accum-steps 16 \
  --num-epochs 2 \
  --num-workers 8 \
  --lr 1.0e-4 \
  --warmup-ratio 0.03 \
  --min-lr-ratio 0.10 \
  --failure-threshold 0.5 \
  --min-class-weight 0.1 \
  --max-class-weight 10
```

输出到：

```text
../mhlc_data/neurons/Qwen3-VL-4B-Instruct/capability/feature_shards/
../mhlc_data/trained_models/neuron_heads/Qwen3-VL-4B-Instruct/capability/neuron_head_final.pt
../mhlc_data/trained_models/neuron_heads/Qwen3-VL-4B-Instruct/capability/final_metrics.json
```

终端会打印 `roc_auc`、`aupr_c`、`aupr_i`、`ece`。

## 8. 探测 Resolution 神经元

这一步全层扫描神经元，用四类标签找和“应该怎么处理问题”最相关的神经元，并保存热力图。

```bash
python src/09_probe_resolution_neurons.py \
  --model-path ../Qwen/Qwen3-VL-4B-Instruct \
  --thinking-mode off \
  --attn-implementation flash_attention_2 \
  --num-workers 4
```

输出到：

```text
../mhlc_data/neurons/Qwen3-VL-4B-Instruct/resolution/
../mhlc_data/visualization/neurons/Qwen3-VL-4B-Instruct/resolution/
```

## 9. 训练 Resolution 神经元 Head

这一步只取探测出的神经元特征训练 3-sigmoid Resolution Head。

```bash
python src/10_train_resolution_neuron_head.py \
  --model-path ../Qwen/Qwen3-VL-4B-Instruct \
  --thinking-mode off \
  --attn-implementation flash_attention_2 \
  --train-batch-size 1 \
  --grad-accum-steps 16 \
  --num-epochs 5 \
  --num-workers 4 \
  --lr 1.0e-4 \
  --warmup-ratio 0.03 \
  --min-lr-ratio 0.10 \
  --decision-threshold 0.5 \
  --min-class-weight 1.0 \
  --max-class-weight 10
```

输出到：

```text
../mhlc_data/neurons/Qwen3-VL-4B-Instruct/resolution/feature_shards/
../mhlc_data/trained_models/neuron_heads/Qwen3-VL-4B-Instruct/resolution/neuron_head_final.pt
../mhlc_data/trained_models/neuron_heads/Qwen3-VL-4B-Instruct/resolution/final_metrics.json
```

终端会打印 `macro_f1` 和 `accuracy`。

## 10. 生成 When2Call 测试 completion

这一步准备 Table3 / Resolution Head 评测需要的测试集 completion。

```bash
python src/06_generate_when2call_eval_completions.py \
  --model-path ../Qwen/Qwen3-VL-4B-Instruct \
  --thinking-mode off \
  --batch-size 64 \
  --max-tokens 16000 \
  --gpu-memory-utilization 0.50 \
  --max-model-len 32000
```

输出到：

```text
../mhlc_data/eval_outputs/when2call/Qwen3-VL-4B-Instruct/when2call_test_generated_4class.parquet
```

## 11. 评测 Capability Head：TriviaQA / Table4

这一步按原项目 TriviaQA 公式计算 `No-Tool`、`Score`、`Calls`、`Precision (%)`、`Missed`。

```bash
python src/11_eval_capability_triviaqa_neuron_head.py \
  --model-path ../Qwen/Qwen3-VL-4B-Instruct \
  --thinking-mode off \
  --attn-implementation flash_attention_3 \
  --thresholds 0.5,0.6,0.7,0.8,0.9,0.95 \
  --generation-batch-size 64 \
  --vllm-max-model-len 32768 \
  --vllm-gpu-memory-utilization 0.40 \
  --vllm-max-num-seqs 128
```

输出到：

```text
../mhlc_data/eval_outputs/neuron_heads/Qwen3-VL-4B-Instruct/capability_triviaqa/
```

关键文件：

```text
summary.json
paper_table4_comparison.json
paper_table4_comparison.csv
threshold_summary.csv
merged_per_example_rows.jsonl
plots/threshold_score_calls.png
```

终端打印三行：`Backbone Choice`、`Backbone + Capability Head（MHLC）`、`Backbone + Capability Head（Ours）`。

## 12. 评测 Resolution Head：When2Call / Table3

这一步按原项目 4 类混淆矩阵口径计算 `F1` 和 `Acc`，显式传 `0.1` 阈值用于对齐原 Table3。

```bash
python src/12_eval_resolution_when2call_neuron_head.py \
  --model-path ../Qwen/Qwen3-VL-4B-Instruct \
  --thinking-mode off \
  --attn-implementation flash_attention_2 \
  --decision-threshold 0.1
```

输出到：

```text
../mhlc_data/eval_outputs/neuron_heads/Qwen3-VL-4B-Instruct/resolution_when2call/
```

关键文件：

```text
summary.json
metrics.json
predictions.jsonl
probability_summary.json
paper_table3_comparison.json
paper_table3_comparison.csv
plots/head_confusion.png
```

终端打印三行：`Backbone Choice`、`Backbone + Resolution Head（MHLC）`、`Backbone + Resolution Head（Ours）`。

## 13. 评测 Capability Head：三项纯文本 benchmark / Table2

这一步跑 `triviaqa`、`math`、`mmlu_pro`，终端打印每项和 `Overall` 的 `score / cost`。`cost` 不会产生新的 API 花费，只按路由到 `model2` 的 token 数乘以你传入的价格参数。

```bash
python src/14_eval_capability_table2_textbench.py \
  --model1-path ../Qwen/Qwen3-VL-4B-Instruct \
  --model2-path ../Qwen/Qwen3-VL-32B-Thinking-FP8 \
  --model1-thinking-mode off \
  --model2-thinking-mode on \
  --baseline-head-path ../mhlc_data/trained_models/baseline_capability_heads/Qwen__Qwen3-VL-4B-Instruct/full/capability_head.pt \
  --attn-implementation flash_attention_3 \
  --thresholds 0.5,0.6,0.7,0.8,0.9 \
  --vllm-max-model-len 32768 \
  --model1-generation-batch-size 128 \
  --model2-generation-batch-size 64 \
  --model1-max-num-seqs 128 \
  --model2-max-num-seqs 64 \
  --model1-gpu-memory-utilization 0.70 \
  --model2-gpu-memory-utilization 0.80 \
  --judge-model-path ../Qwen/Qwen3-VL-8B-Instruct \
  --judge-batch-size 32 \
  --m2-input-cost-per-1m-usd 0.70 \
  --m2-output-cost-per-1m-usd 8.40
```

输出到：

```text
../mhlc_data/eval_outputs/neuron_heads/Qwen3-VL-4B-Instruct/capability_table2_textbench/
```

关键文件：

```text
run_config.json
table2_textbench_comparison.json
table2_textbench_comparison.csv
table2_textbench_comparison_threshold_0p5.json
table2_textbench_comparison_threshold_0p5.csv
plots/table2_score_cost.png
plots/table2_score_cost_threshold_0p5.png
triviaqa/single_agent_model1/results.jsonl
triviaqa/single_agent_model2/results.jsonl
triviaqa/head_scores_mhlc.jsonl
triviaqa/head_scores_ours.jsonl
triviaqa/routed_mhlc_threshold_0p5/results_scored.jsonl
triviaqa/routed_ours_threshold_0p5/results_scored.jsonl
```

`math/` 和 `mmlu_pro/` 下也会保存同样结构的文件。终端会按 `--thresholds` 中的每个阈值分别打印一张 Table2 风格表格。

正式命令已经启用原项目 Table2 评测口径的 8B judge：

```bash
--judge-model-path ../Qwen/Qwen3-VL-8B-Instruct
```

如果只想快速规则评测，可以把 `--judge-model-path` 留空，但那不是正式口径。

如果 baseline head 是原项目自己训练保存的旧文件名，可以把参数改成：

```bash
--baseline-head-path ../some_dir/aux_head_final.pt
```

## 续跑和重跑

默认会复用已有数据、生成结果、head 分数和评测结果。想重跑某一步，在对应命令最后加：

```bash
--clean
```

Table2 想单独控制复用项，可以加：

```bash
--no-reuse-generations
--no-reuse-scores
--no-reuse-evaluations
```

所有可视化都保存为本地图片，不使用 wandb。
