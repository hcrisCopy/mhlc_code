"""Small helpers for one-node, one-process-per-GPU MHLC runs.

The formal multi-GPU mode deliberately does not use tensor parallelism.  Each
``torchrun`` worker sees one physical GPU, loads an unchanged copy of the
backbone, and receives a disjoint deterministic data partition.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ParallelContext:
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        """Synchronize torchrun workers only when parallel mode is active."""
        if not self.enabled:
            return
        import torch.distributed as dist

        if not dist.is_initialized():
            # Gloo is sufficient here: GPU work is independent and only file
            # production / CPU-side aggregation is synchronized.
            dist.init_process_group(backend="gloo", init_method="env://")
        dist.barrier()


def configure_parallel_context() -> ParallelContext:
    """Read torchrun environment and expose exactly one GPU to each worker.

    Call this before importing torch, vLLM, or transformers.  A pre-set
    CUDA_VISIBLE_DEVICES list is respected by selecting its local-rank entry.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    ctx = ParallelContext(rank=rank, world_size=world_size, local_rank=local_rank)
    if not ctx.enabled:
        return ctx

    visible = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
    if len(visible) > local_rank:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible[local_rank]
    elif len(visible) != 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
    return ctx


def contiguous_range(total: int, ctx: ParallelContext) -> tuple[int, int]:
    """Return the deterministic contiguous part assigned to this worker."""
    total = max(0, int(total))
    start = total * ctx.rank // ctx.world_size
    end = total * (ctx.rank + 1) // ctx.world_size
    return start, end


def worker_dir(base: Path, stage: str, ctx: ParallelContext) -> Path:
    """Stable sidecar directory, intentionally outside a consumer data tree."""
    safe_stage = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stage)
    return base.parent / f".{base.name}.{safe_stage}.eight_gpu" / f"rank_{ctx.rank:02d}"


def require_single_gpu_vllm(tensor_parallel_size: int, ctx: ParallelContext) -> None:
    if ctx.enabled and int(tensor_parallel_size) != 1:
        raise ValueError(
            "One-node multi-GPU mode uses one complete model per GPU; "
            "set --tensor-parallel-size 1."
        )
