# Table2 纯文本 Capability Head 评测

这个评测脚本对齐原项目 Table2 的纯文本 benchmark 流程，跑：

```text
triviaqa
math
mmlu_pro
```

终端会打印类似 Table2 的 `score / cost` 表，并额外给 `Overall`。这里的 `cost` 不是重新调用付费 API，而是按原项目思路：只统计被路由到 `model2` 的输入/输出 token，再乘以你传入的价格参数。`model1` 和两个 head 都按本地免费算。

## 需要已有产物

benchmark 数据：

```text
../mhlc_data/data/benchmarks/triviaqa/dataset
../mhlc_data/data/benchmarks/merged_math.csv
../mhlc_data/data/benchmarks/test.csv
```

原作者 Capability Head：

```text
../mhlc_data/trained_models/baseline_capability_heads/Qwen__Qwen3-VL-4B-Instruct/full/capability_head.pt
```

我们的神经元 Capability Head：

```text
../mhlc_data/trained_models/neuron_heads/Qwen3-VL-4B-Instruct/capability/neuron_head_final.pt
../mhlc_data/neurons/Qwen3-VL-4B-Instruct/capability/selected_neurons.jsonl
```

如果 benchmark 数据还没准备好，先跑：

```bash
python src/01_download_data.py --group benchmarks --benchmarks triviaqa,math,mmlu_pro
```

如果你有论文/作者快照版的两个 CSV，优先导入它们：

```bash
python src/01_download_data.py --group benchmarks \
  --benchmarks math,mmlu_pro \
  --math-csv ../paper_snapshots/merged_math.csv \
  --mmlu-pro-csv ../paper_snapshots/test.csv
```

## 正式评测

从项目根目录运行：

```bash
python src/14_eval_capability_table2_textbench.py \
  --model1-path ../Qwen/Qwen3-VL-4B-Instruct \
  --model2-path ../Qwen/Qwen3-VL-32B-Thinking-FP8 \
  --baseline-head-path ../mhlc_data/trained_models/baseline_capability_heads/Qwen__Qwen3-VL-4B-Instruct/full/capability_head.pt \
  --attn-implementation sdpa \
  --threshold 0.8 \
  --m2-input-cost-per-1m-usd 0.70 \
  --m2-output-cost-per-1m-usd 8.40
```

默认跑三项 benchmark，每项最多 1000 条。终端打印四行：

```text
Backbone Choice
Always Call Strong Model
Backbone + Capability Head（MHLC）
Backbone + Capability Head（Ours）
```

其中后两行就是原作者 hidden states head 和我们的探测神经元 head 的直接对比。

## 4090 单卡 smoke

这个命令只为跑通流程。它把 `model2` 临时也设成 4B，并且每个 benchmark 只取 2 条：

```bash
python src/14_eval_capability_table2_textbench.py \
  --model1-path ../Qwen/Qwen3-VL-4B-Instruct \
  --model2-path ../Qwen/Qwen3-VL-4B-Instruct \
  --model2-thinking-mode off \
  --baseline-head-path ../mhlc_data/trained_models/baseline_capability_heads/Qwen__Qwen3-VL-4B-Instruct/full/capability_head.pt \
  --attn-implementation sdpa \
  --max-samples 2 \
  --vllm-max-model-len 16000 \
  --max-seq-len 16000 \
  --model1-generation-batch-size 1 \
  --model2-generation-batch-size 1 \
  --model1-max-num-seqs 1 \
  --model2-max-num-seqs 1 \
  --score-batch-size 1 \
  --model1-gpu-memory-utilization 0.70 \
  --model2-gpu-memory-utilization 0.70 \
  --output-dir ../mhlc_data/eval_outputs/neuron_heads/Qwen3-VL-4B-Instruct/capability_table2_textbench_smoke
```

如果只想更快确认单个数据集，可以加：

```bash
--benchmarks triviaqa
```

如果仍然在 vLLM CUDA graph 初始化阶段显存不够，再额外加：

```bash
--vllm-enforce-eager
```

如果你用的是原项目自己训练保存的文件名，也可以传：

```bash
--baseline-head-path ../some_dir/aux_head_final.pt
```

## 输出位置

正式默认输出：

```text
../mhlc_data/eval_outputs/neuron_heads/Qwen3-VL-4B-Instruct/capability_table2_textbench/
```

关键文件：

```text
run_config.json
table2_textbench_comparison.json
table2_textbench_comparison.csv
plots/table2_score_cost.png
triviaqa/single_agent_model1/results.jsonl
triviaqa/single_agent_model2/results.jsonl
triviaqa/head_scores_mhlc.jsonl
triviaqa/head_scores_ours.jsonl
triviaqa/routed_mhlc/results_scored.jsonl
triviaqa/routed_ours/results_scored.jsonl
```

`math/` 和 `mmlu_pro/` 下也会保存同样结构的文件。

## 续跑和清理

默认会复用已经生成的结果：

```text
--reuse-generations
--reuse-scores
--reuse-evaluations
```

想重跑旧输出，加：

```bash
--clean
```

想强制重新生成某一类中间产物，可以用：

```bash
--no-reuse-generations
--no-reuse-scores
--no-reuse-evaluations
```

## 计算口径

`score` 使用原项目 evaluator 对 `triviaqa/math/mmlu_pro` 的 `accuracy`。

`cost` 公式：

```text
model2_prompt_tokens * input_price / 1000000
+ model2_completion_tokens * output_price / 1000000
```

`Overall score` 是三个 benchmark score 的平均值，`Overall cost` 是三个 benchmark cost 的和。

可视化保存为本地图片，不使用 wandb。
