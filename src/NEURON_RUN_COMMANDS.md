# 神经元 Head 运行命令

从项目根目录运行：

```bash
cd mhlc_code
```

默认模型是：

```text
../Qwen/Qwen3-VL-4B-Instruct
```

正式实验只需要把 `--model-path` 换成目标模型路径；其他参数默认按正式设置跑全量数据。

## 1. Capability 神经元探测

```bash
python src/07_probe_capability_neurons.py \
  --model-path ../Qwen/Qwen3-VL-4B-Instruct
```

输出：

```text
../mhlc_data/neurons/Qwen3-VL-4B-Instruct/capability/
../mhlc_data/visualization/neurons/Qwen3-VL-4B-Instruct/capability/
```

## 2. Capability 神经元 head 训练

```bash
python src/08_train_capability_neuron_head.py \
  --model-path ../Qwen/Qwen3-VL-4B-Instruct
```

输出：

```text
../mhlc_data/neurons/Qwen3-VL-4B-Instruct/capability/feature_shards/
../mhlc_data/trained_models/neuron_heads/Qwen3-VL-4B-Instruct/capability/
```

最终 checkpoint：

```text
../mhlc_data/trained_models/neuron_heads/Qwen3-VL-4B-Instruct/capability/neuron_head_final.pt
```

## 3. Resolution 神经元探测

```bash
python src/09_probe_resolution_neurons.py \
  --model-path ../Qwen/Qwen3-VL-4B-Instruct
```

输出：

```text
../mhlc_data/neurons/Qwen3-VL-4B-Instruct/resolution/
../mhlc_data/visualization/neurons/Qwen3-VL-4B-Instruct/resolution/
```

## 4. Resolution 神经元 head 训练

```bash
python src/10_train_resolution_neuron_head.py \
  --model-path ../Qwen/Qwen3-VL-4B-Instruct
```

输出：

```text
../mhlc_data/neurons/Qwen3-VL-4B-Instruct/resolution/feature_shards/
../mhlc_data/trained_models/neuron_heads/Qwen3-VL-4B-Instruct/resolution/
```

最终 checkpoint：

```text
../mhlc_data/trained_models/neuron_heads/Qwen3-VL-4B-Instruct/resolution/neuron_head_final.pt
```

## 跑通小样本

只想先确认流程时，加 `--max-samples`：

```bash
python src/07_probe_capability_neurons.py \
  --model-path ../Qwen/Qwen3-VL-4B-Instruct \
  --max-samples 64

python src/08_train_capability_neuron_head.py \
  --model-path ../Qwen/Qwen3-VL-4B-Instruct \
  --max-samples 64
```

Resolution 同理：

```bash
python src/09_probe_resolution_neurons.py \
  --model-path ../Qwen/Qwen3-VL-4B-Instruct \
  --max-samples 64

python src/10_train_resolution_neuron_head.py \
  --model-path ../Qwen/Qwen3-VL-4B-Instruct \
  --max-samples 64
```

## 续跑和清理

默认会按 manifest 复用已有产物：

```text
探测产物存在且参数一致 -> 跳过
feature shard 已存在且大小正确 -> 跳过
训练 checkpoint 存在 -> 默认续训
```

需要重跑某一阶段，在命令后加：

```bash
--clean
```

例如：

```bash
python src/07_probe_capability_neurons.py \
  --model-path ../Qwen/Qwen3-VL-4B-Instruct \
  --clean
```

