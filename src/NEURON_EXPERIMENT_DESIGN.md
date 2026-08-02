# MHLC 神经元 Head 实验设计

本方案不修改 `Multi-Head-Latent-Control/`。前面的数据准备阶段仍然负责生成 completion 和标签；这里从这些已准备好的数据继续做全层 FFN 神经元探测，再用选出的神经元训练两个 head。

## 1. 接的数据

Capability 读：

```text
../mhlc_data/data/train/Qwen3VL/Qwen3_VL_4B_Instruct_text_only_OriginalMixedShare_40851/verified/
```

这里的 `correctness_score` 来自前面 judge 模型评判。神经元探测不重新 judge，只消费这个分数。

Resolution / When2Call 读：

```text
../mhlc_data/data/train/when2call/qwen3vl/Qwen3-VL-4B-Instruct_4class/
```

这里的 4 类行为标签来自前面的 annotator / label 构造阶段。训练仍沿用原项目 3-sigmoid 口径：

```text
tool_call        -> [1, 0, 0]
request_for_info -> [0, 1, 0]
cannot_answer    -> [0, 0, 1]
direct_answer    -> [0, 0, 0]
```

## 2. 神经元定义

参考 TKN，神经元定义为每层 MLP `down_proj` 输入处的 FFN intermediate 坐标：

```text
neuron = (layer, neuron_index)
h = act(gate_proj(x)) * up_proj(x)
```

代码通过 `down_proj` 的 forward pre-hook 抓取 `h`。为了和 MHLC 原训练保持一致，只使用 `completion_text_only` token，不使用 prompt、模板控制 token 或视觉 token。

每个样本对一个神经元的激活值定义为 completion token 上的平均值：

```text
a_i(layer, j) = mean_t h_i,t,layer,j
```

默认全层探测，不预设 Capability 在后层、Resolution 在中层。层分布只作为结果分析。

## 3. Capability 打分

任务本质：判断当前小模型回答是否可靠，是否应该交给强模型。

标签：

```text
y_i = correctness_score_i
```

默认分组：

```text
H = {i | y_i >= 0.8}
L = {i | y_i < 0.5}
```

如果某组样本太少，就退化成 top 30% / bottom 30%。

对每个神经元计算：

```text
delta = mean(a_H) - mean(a_L)
separation = abs(delta) / pooled_std
correlation = abs(corr(a_i, y_i))
responsiveness = abs(delta)
```

默认还乘上 `down_proj` 列范数的归一化因子，让“下游影响更强”的 FFN intermediate 神经元略微加权：

```text
weighted_separation = separation * norm_factor
weighted_responsiveness = responsiveness * norm_factor
```

最终分数：

```text
capability_score =
  relu_z(weighted_separation)
  + relu_z(abs(correlation))
  + 0.5 * relu_z(weighted_responsiveness)
```

其中 `relu_z(x) = max(0, zscore_within_layer(x))`。

方向含义：

```text
delta > 0 -> correct_high
delta < 0 -> failure_high
```

因此 Capability 神经元的意义是：它们的 completion 激活和“回答可靠 / 回答失败风险”最有判别关系。

## 4. Resolution 打分

任务本质：判断当前回答是否需要工具、澄清、拒答，还是直接回答。

对三个显式类分别打分：

```text
c in {tool_call, request_for_info, cannot_answer}
P_c = {i | target_i,c = 1}
N_c = {i | target_i,c = 0}
```

每个类的神经元分数：

```text
delta_c = mean(a_Pc) - mean(a_Nc)
separation_c = abs(delta_c) / pooled_std_c
responsiveness_c = abs(delta_c)
class_score_c =
  relu_z(weighted_separation_c)
  + 0.5 * relu_z(weighted_responsiveness_c)
```

合并分数：

```text
resolution_score = max(class_score_tool, class_score_info, class_score_cant)
best_class = argmax(...)
```

因此 Resolution 神经元的意义是：它们最能区分某一种干预行为和其他行为。

## 5. 神经元数量限制

每层最多选：

```text
floor(0.10 * layer_ffn_dim)
```

全局再检查：

```text
selected_total <= floor(0.10 * total_ffn_dim)
```

所以选中神经元不会超过总 FFN intermediate 神经元的 10%。

## 6. 训练 head

训练阶段不再送 hidden states。流程是：

```text
选中神经元 -> 重新跑一遍数据 -> 只缓存这些神经元的 mean activation -> 训练小 MLP head
```

Capability head：

```text
input  = selected neuron activations
target = correctness_score
loss   = failure-aware weighted MSE
```

Resolution head：

```text
input  = selected neuron activations
target = 3-sigmoid labels
loss   = multi-label BCEWithLogits
```

特征按 shard 保存，训练中断后可以继续补缺失 shard；训练 checkpoint 保存 optimizer 和进度，默认续训。

## 7. 输出

Capability 默认输出：

```text
../mhlc_data/neurons/Qwen3-VL-4B-Instruct/capability/
../mhlc_data/visualization/neurons/Qwen3-VL-4B-Instruct/capability/
../mhlc_data/trained_models/neuron_heads/Qwen3-VL-4B-Instruct/capability/
```

Resolution 默认输出：

```text
../mhlc_data/neurons/Qwen3-VL-4B-Instruct/resolution/
../mhlc_data/visualization/neurons/Qwen3-VL-4B-Instruct/resolution/
../mhlc_data/trained_models/neuron_heads/Qwen3-VL-4B-Instruct/resolution/
```

主要文件：

```text
selected_neurons.jsonl
neuron_scores.pt
layer_summary.csv
layer_top_neuron_score_heatmap.png
selected_density_by_layer.png
neuron_head_final.pt
```

