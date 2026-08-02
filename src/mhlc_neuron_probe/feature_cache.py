from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .baseline_bridge import build_forward_inputs, load_frozen_backbone, move_batch_to_device
from .data_tasks import (
    CapabilityDataConfig,
    ResolutionDataConfig,
    capability_collator,
    load_capability_dataset,
    load_resolution_dataset,
    resolution_collator,
)
from .ffn_hooks import FFNActivationCollector, discover_ffn_intermediate_layers, down_weight_norms, module_meta_from_layers
from .io_utils import read_json, read_jsonl, rel, stable_hash, write_json


@dataclass
class FeaturePlan:
    rows: list[dict[str, Any]]
    groups: dict[str, dict[str, Any]]
    num_features: int
    selected_hash: str


def build_feature_plan(neuron_rows: list[dict[str, Any]]) -> FeaturePlan:
    rows = sorted(neuron_rows, key=lambda row: int(row["rank"]))
    groups: dict[str, dict[str, Any]] = {}
    for position, row in enumerate(rows):
        key = str(row["module_key"])
        groups.setdefault(key, {"positions": [], "indices": []})
        groups[key]["positions"].append(position)
        groups[key]["indices"].append(int(row["index"]))
    for group in groups.values():
        group["positions_tensor"] = torch.tensor(group["positions"], dtype=torch.long)
        group["indices_tensor"] = torch.tensor(group["indices"], dtype=torch.long)
    return FeaturePlan(
        rows=rows,
        groups=groups,
        num_features=len(rows),
        selected_hash=stable_hash([{"module_key": row["module_key"], "index": row["index"], "rank": row["rank"]} for row in rows]),
    )


def selected_features_from_captures(captures: dict[str, torch.Tensor], plan: FeaturePlan) -> torch.Tensor:
    if plan.num_features <= 0:
        raise ValueError("No selected neurons are available for feature extraction.")
    batch_size = next(iter(captures.values())).shape[0]
    features = torch.empty(batch_size, plan.num_features, dtype=torch.float32)
    for key, group in plan.groups.items():
        block = captures[key].float().index_select(1, group["indices_tensor"])
        features[:, group["positions_tensor"]] = block
    return features


def _feature_cache_params(args: Any, *, task: str, dataset_len: int, plan: FeaturePlan) -> dict[str, Any]:
    return {
        "stage": "neuron_feature_cache",
        "task": task,
        "model_path": args.model_path,
        "dataset_path": args.dataset_path,
        "dataset_len": int(dataset_len),
        "selected_hash": plan.selected_hash,
        "feature_pooling": "completion_text_only_mean",
        "feature_shard_size": int(args.feature_shard_size),
        "batch_size": int(args.extract_batch_size),
        "max_seq_len": int(args.max_seq_len),
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "seed": int(args.seed),
        "max_samples": int(args.max_samples),
    }


def _valid_shard(path: Path, expected_count: int, task: str) -> bool:
    if not path.exists():
        return False
    try:
        payload = torch.load(path, map_location="cpu")
    except Exception:
        return False
    if "features" not in payload or int(payload["features"].shape[0]) != int(expected_count):
        return False
    if task == "capability":
        return "labels" in payload
    return "targets" in payload and "usable_mask" in payload and "behavior_ids" in payload


def ensure_feature_cache(args: Any, *, task: str, neuron_path: Path, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    neuron_rows = read_jsonl(neuron_path)
    plan = build_feature_plan(neuron_rows)

    if task == "capability":
        data_cfg = CapabilityDataConfig(
            dataset_path=args.dataset_path,
            aux_label_column=args.aux_label_column,
            subset_name=args.subset_name,
            seed=int(args.seed),
            max_samples=int(args.max_samples),
            max_seq_len=int(args.max_seq_len),
            head_input_mode=args.head_input_mode,
        )
        dataset = load_capability_dataset(data_cfg)
        collator_factory = lambda processor: capability_collator(processor, data_cfg)
    else:
        data_cfg = ResolutionDataConfig(
            dataset_path=args.dataset_path,
            subset_name=args.subset_name,
            seed=int(args.seed),
            max_samples=int(args.max_samples),
            max_seq_len=int(args.max_seq_len),
            head_input_mode=args.head_input_mode,
            max_head_input_tokens=args.max_head_input_tokens,
            drop_unusable_rows=True,
        )
        dataset = load_resolution_dataset(data_cfg)
        collator_factory = lambda processor: resolution_collator(processor, data_cfg)

    params = _feature_cache_params(args, task=task, dataset_len=len(dataset), plan=plan)
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest.get("params") == params and all(Path(item["path"]).exists() for item in manifest.get("shards", [])):
            print(f"[skip] feature cache already complete: {rel(cache_dir)}", flush=True)
            return manifest_path

    runtime = load_frozen_backbone(
        model_name_or_path=args.model_path,
        model_family=args.model_family,
        thinking_mode=args.thinking_mode,
        trust_remote_code=bool(args.trust_remote_code),
        attn_implementation=args.attn_implementation,
        prefer_unsloth_mirror=bool(args.prefer_unsloth_mirror),
        load_in_4bit=bool(args.load_in_4bit),
        load_in_8bit=bool(args.load_in_8bit),
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        dtype=args.dtype,
        max_seq_len=int(args.max_seq_len),
        max_pixels=int(args.max_pixels),
    )
    layers = discover_ffn_intermediate_layers(runtime.forward_model)
    module_meta = module_meta_from_layers(layers)
    _ = down_weight_norms(layers)
    collator = collator_factory(runtime.processor)
    shard_size = int(args.feature_shard_size)
    num_shards = math.ceil(len(dataset) / shard_size)
    shards: list[dict[str, Any]] = []
    save_dtype = torch.float16 if str(args.feature_save_dtype).lower() == "float16" else torch.float32

    with FFNActivationCollector(layers, save_dtype=torch.float32) as collector:
        for shard_id in range(num_shards):
            start = shard_id * shard_size
            end = min(len(dataset), start + shard_size)
            shard_path = cache_dir / f"shard_{shard_id:05d}.pt"
            expected_count = end - start
            if _valid_shard(shard_path, expected_count, task):
                print(f"[skip] feature shard exists: {rel(shard_path)}", flush=True)
                shards.append({"path": str(shard_path), "start": start, "end": end, "count": expected_count})
                continue

            shard_ds = dataset.select(range(start, end))
            loader = DataLoader(
                shard_ds,
                batch_size=int(args.extract_batch_size),
                shuffle=False,
                collate_fn=collator,
                num_workers=int(args.num_workers),
                pin_memory=bool(args.pin_memory and torch.cuda.is_available()),
                drop_last=False,
            )
            feature_parts: list[torch.Tensor] = []
            label_parts: list[torch.Tensor] = []
            target_parts: list[torch.Tensor] = []
            usable_parts: list[torch.Tensor] = []
            behavior_parts: list[torch.Tensor] = []
            for batch in tqdm(loader, desc=f"features {task} shard {shard_id + 1}/{num_shards}", dynamic_ncols=True):
                batch = move_batch_to_device(batch, runtime.device, runtime.fp_dtype)
                token_mask = batch.pop("head_token_mask")
                labels = batch.pop("aux_labels", None)
                targets = batch.pop("aux_targets", None)
                usable = batch.pop("aux_usable_mask", None)
                behavior_ids = batch.pop("aux_behavior_ids", None)
                collector.set_token_mask(token_mask)
                collector.clear()
                forward_inputs = build_forward_inputs(batch, runtime.model_family)
                autocast_enabled = runtime.device.type == "cuda" and runtime.fp_dtype in {torch.float16, torch.bfloat16}
                autocast_dtype = runtime.fp_dtype if runtime.fp_dtype is not None else torch.bfloat16
                with torch.no_grad():
                    with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=autocast_enabled):
                        _ = runtime.forward_model(**forward_inputs, use_cache=False, return_dict=True)
                missing = [meta["key"] for meta in module_meta if meta["key"] not in collector.captures]
                if missing:
                    raise RuntimeError(f"Missing FFN captures: {missing[:3]}")
                feature_parts.append(selected_features_from_captures(collector.captures, plan).to(save_dtype))
                if labels is not None:
                    label_parts.append(labels.detach().cpu().float())
                if targets is not None:
                    target_parts.append(targets.detach().cpu().float())
                if usable is not None:
                    usable_parts.append(usable.detach().cpu().float())
                if behavior_ids is not None:
                    behavior_parts.append(behavior_ids.detach().cpu().long())

            payload: dict[str, Any] = {
                "features": torch.cat(feature_parts, dim=0).contiguous(),
                "start": start,
                "end": end,
                "selected_hash": plan.selected_hash,
            }
            if task == "capability":
                payload["labels"] = torch.cat(label_parts, dim=0).contiguous()
            else:
                payload["targets"] = torch.cat(target_parts, dim=0).contiguous()
                payload["usable_mask"] = torch.cat(usable_parts, dim=0).contiguous()
                payload["behavior_ids"] = torch.cat(behavior_parts, dim=0).contiguous()
            torch.save(payload, shard_path)
            shards.append({"path": str(shard_path), "start": start, "end": end, "count": expected_count})
            print(f"[write] {rel(shard_path)}", flush=True)

    summary = {
        "task": task,
        "rows": len(dataset),
        "num_features": plan.num_features,
        "num_shards": len(shards),
        "feature_save_dtype": args.feature_save_dtype,
    }
    write_json(manifest_path, {"params": params, "summary": summary, "shards": shards})
    compute_feature_norm(manifest_path)
    return manifest_path


def compute_feature_norm(manifest_path: Path) -> Path:
    norm_path = manifest_path.parent / "feature_norm.pt"
    if norm_path.exists():
        return norm_path
    manifest = read_json(manifest_path)
    sum_x: torch.Tensor | None = None
    sum_x2: torch.Tensor | None = None
    n = 0
    for shard in tqdm(manifest["shards"], desc="feature norm", dynamic_ncols=True):
        payload = torch.load(shard["path"], map_location="cpu")
        x = payload["features"].float()
        if sum_x is None:
            sum_x = torch.zeros(x.shape[1], dtype=torch.float64)
            sum_x2 = torch.zeros(x.shape[1], dtype=torch.float64)
        sum_x += x.double().sum(dim=0)
        sum_x2 += torch.square(x.double()).sum(dim=0)
        n += int(x.shape[0])
    mean = (sum_x / max(n, 1)).float()
    var = (sum_x2 / max(n, 1) - mean.double().square()).clamp_min(1.0e-6).float()
    torch.save({"mean": mean, "std": torch.sqrt(var), "n": n}, norm_path)
    return norm_path


class ShardedFeatureDataset(Dataset):
    def __init__(self, manifest_path: Path, task: str):
        self.manifest = read_json(manifest_path)
        self.task = task
        self.shards = self.manifest["shards"]
        self.ends = [int(item["end"]) for item in self.shards]
        self.total = int(self.shards[-1]["end"]) if self.shards else 0
        self._cache_index = -1
        self._cache_payload: dict[str, torch.Tensor] | None = None

    def __len__(self) -> int:
        return self.total

    def _load_shard(self, shard_index: int) -> dict[str, torch.Tensor]:
        if self._cache_index != shard_index or self._cache_payload is None:
            self._cache_payload = torch.load(self.shards[shard_index]["path"], map_location="cpu")
            self._cache_index = shard_index
        return self._cache_payload

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        shard_index = bisect.bisect_right(self.ends, int(index))
        shard = self.shards[shard_index]
        local = int(index) - int(shard["start"])
        payload = self._load_shard(shard_index)
        item = {"features": payload["features"][local].float()}
        if self.task == "capability":
            item["labels"] = payload["labels"][local].float()
        else:
            item["targets"] = payload["targets"][local].float()
            item["usable_mask"] = payload["usable_mask"][local].float()
            item["behavior_ids"] = payload["behavior_ids"][local].long()
        return item

