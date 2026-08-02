#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from mhlc_data_prep.paths import (
    ensure_mhlc_data_layout,
    resolve_from_code_root,
    set_hf_dirs_inside_data_root,
)
from mhlc_data_prep.run_utils import rel


@dataclass(frozen=True)
class CapabilityHeadSpec:
    key: str
    base_model: str
    repo_id: str
    variant: str
    thinking_mode: str
    description: str
    aliases: tuple[str, ...]


CAPABILITY_HEADS: tuple[CapabilityHeadSpec, ...] = (
    CapabilityHeadSpec(
        key="qwen3vl-2b-thinking-full",
        base_model="Qwen/Qwen3-VL-2B-Thinking",
        repo_id="AmirhoseinGH/mhlc-capability-head-qwen3vl-2b-thinking",
        variant="full",
        thinking_mode="on",
        description="Qwen3-VL-2B Thinking, full trajectory",
        aliases=("Qwen3-VL-2B-Thinking", "qwen3vl-2b-thinking"),
    ),
    CapabilityHeadSpec(
        key="qwen3vl-4b-instruct-full",
        base_model="Qwen/Qwen3-VL-4B-Instruct",
        repo_id="AmirhoseinGH/mhlc-capability-head-qwen3vl-4b-instruct",
        variant="full",
        thinking_mode="off",
        description="Qwen3-VL-4B Instruct, full trajectory",
        aliases=("Qwen3-VL-4B-Instruct", "qwen3vl-4b-instruct"),
    ),
    CapabilityHeadSpec(
        key="qwen3vl-4b-thinking-full",
        base_model="Qwen/Qwen3-VL-4B-Thinking",
        repo_id="AmirhoseinGH/mhlc-capability-head-qwen3vl-4b-thinking",
        variant="full",
        thinking_mode="on",
        description="Qwen3-VL-4B Thinking, full trajectory",
        aliases=("Qwen3-VL-4B-Thinking", "qwen3vl-4b-thinking"),
    ),
    CapabilityHeadSpec(
        key="qwen35-4b-full",
        base_model="Qwen/Qwen3.5-4B",
        repo_id="AmirhoseinGH/mhlc-capability-head-qwen35-4b",
        variant="full",
        thinking_mode="off",
        description="Qwen3.5-4B, thinking off, full trajectory",
        aliases=("Qwen3.5-4B", "Qwen3_5_4B", "qwen35-4b"),
    ),
    CapabilityHeadSpec(
        key="qwen35-9b-full",
        base_model="Qwen/Qwen3.5-9B",
        repo_id="AmirhoseinGH/mhlc-capability-head-qwen35-9b",
        variant="full",
        thinking_mode="off",
        description="Qwen3.5-9B, thinking off, full trajectory",
        aliases=("Qwen3.5-9B", "Qwen3_5_9B", "qwen35-9b"),
    ),
    CapabilityHeadSpec(
        key="gemma4-e4b-instruct-full",
        base_model="google/gemma-4-E4B-it",
        repo_id="AmirhoseinGH/mhlc-capability-head-gemma4-e4b-instruct",
        variant="full",
        thinking_mode="off",
        description="Gemma 4 E4B it, instruct/full trajectory",
        aliases=("gemma-4-E4B-it", "gemma4-e4b-it", "gemma4-e4b-instruct"),
    ),
    CapabilityHeadSpec(
        key="gemma4-e4b-thinking-full",
        base_model="google/gemma-4-E4B-it",
        repo_id="AmirhoseinGH/mhlc-capability-head-gemma4-e4b-thinking",
        variant="full",
        thinking_mode="on",
        description="Gemma 4 E4B it, thinking/full trajectory",
        aliases=("gemma-4-E4B-it", "gemma4-e4b-it", "gemma4-e4b-thinking"),
    ),
    CapabilityHeadSpec(
        key="qwen3vl-32b-instruct-full",
        base_model="Qwen/Qwen3-VL-32B-Instruct-FP8",
        repo_id="AmirhoseinGH/mhlc-capability-head-qwen3vl-32b-instruct-step10000",
        variant="full",
        thinking_mode="off",
        description="Qwen3-VL-32B Instruct FP8, full trajectory",
        aliases=("Qwen3-VL-32B-Instruct-FP8", "qwen3vl-32b-instruct"),
    ),
    CapabilityHeadSpec(
        key="qwen3vl-4b-thinking-lite",
        base_model="Qwen/Qwen3-VL-4B-Thinking",
        repo_id="AmirhoseinGH/mhlc-capability-head-qwen3vl-4b-thinking-lite",
        variant="lite",
        thinking_mode="on",
        description="Qwen3-VL-4B Thinking, lightweight full trajectory",
        aliases=("Qwen3-VL-4B-Thinking", "qwen3vl-4b-thinking-lite"),
    ),
    CapabilityHeadSpec(
        key="qwen3vl-2b-thinking-prefix200",
        base_model="Qwen/Qwen3-VL-2B-Thinking",
        repo_id="AmirhoseinGH/mhlc-capability-head-qwen3vl-2b-thinking-prefix200",
        variant="prefix200",
        thinking_mode="on",
        description="Qwen3-VL-2B Thinking, first 200 completion tokens",
        aliases=("Qwen3-VL-2B-Thinking", "qwen3vl-2b-thinking-prefix200"),
    ),
    CapabilityHeadSpec(
        key="qwen3vl-4b-thinking-prefix200",
        base_model="Qwen/Qwen3-VL-4B-Thinking",
        repo_id="AmirhoseinGH/mhlc-capability-head-qwen3vl-4b-thinking-prefix200",
        variant="prefix200",
        thinking_mode="on",
        description="Qwen3-VL-4B Thinking, first 200 completion tokens",
        aliases=("Qwen3-VL-4B-Thinking", "qwen3vl-4b-thinking-prefix200"),
    ),
    CapabilityHeadSpec(
        key="gemma4-e4b-thinking-prefix200",
        base_model="google/gemma-4-E4B-it",
        repo_id="AmirhoseinGH/mhlc-capability-head-gemma4-e4b-thinking-prefix200",
        variant="prefix200",
        thinking_mode="on",
        description="Gemma 4 E4B it, thinking, first 200 completion tokens",
        aliases=("gemma-4-E4B-it", "gemma4-e4b-thinking-prefix200"),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download released MHLC baseline Capability Head weights into ../mhlc_data."
    )
    parser.add_argument("--model", default=None, help="Backbone model id or local model path.")
    parser.add_argument("--repo-id", default=None, help="Explicit Hugging Face head repo id. Overrides model matching.")
    parser.add_argument("--all", action="store_true", help="Download every known released Capability Head.")
    parser.add_argument("--data-root", default="../mhlc_data")
    parser.add_argument(
        "--variant",
        default="auto",
        choices=["auto", "full", "lite", "prefix200"],
        help="Head variant. Auto prefers the full trajectory head when multiple heads match.",
    )
    parser.add_argument(
        "--thinking-mode",
        default="auto",
        choices=["auto", "on", "off"],
        help="Disambiguates backbones that have both instruct and thinking heads.",
    )
    parser.add_argument("--output-dir", default=None, help="Optional explicit output directory.")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--token", default=None, help="Optional Hugging Face token.")
    parser.add_argument("--force", action="store_true", help="Force re-download from Hugging Face.")
    parser.add_argument("--no-readme", action="store_true", help="Do not download the model card README.md.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve the repo and output path without downloading.")
    parser.add_argument("--list", action="store_true", help="List known released Capability Head mappings and exit.")
    return parser.parse_args()


def normalize_text(value: str | Path | None) -> str:
    if value is None:
        return ""
    text = str(value).strip().replace("\\", "/").rstrip("/")
    text = re.sub(r"^.*?/Qwen/", "Qwen/", text, flags=re.IGNORECASE)
    text = re.sub(r"^.*?/google/", "google/", text, flags=re.IGNORECASE)
    return text.lower()


def model_basename(value: str | Path | None) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    return text.rsplit("/", 1)[-1]


def spec_tokens(spec: CapabilityHeadSpec) -> set[str]:
    values = {spec.base_model, model_basename(spec.base_model), spec.key}
    values.update(spec.aliases)
    values.update(model_basename(alias) for alias in spec.aliases)
    return {normalize_text(v) for v in values if normalize_text(v)}


def matches_model(spec: CapabilityHeadSpec, model: str) -> bool:
    target = normalize_text(model)
    target_base = model_basename(model)
    if not target:
        return False
    tokens = spec_tokens(spec)
    return target in tokens or target_base in tokens


def choose_spec(model: str, variant: str, thinking_mode: str) -> CapabilityHeadSpec:
    matches = [spec for spec in CAPABILITY_HEADS if matches_model(spec, model)]
    if not matches:
        known = "\n".join(f"  - {spec.base_model} ({spec.variant}, thinking={spec.thinking_mode})" for spec in CAPABILITY_HEADS)
        raise SystemExit(f"No released Capability Head mapping found for model={model!r}.\nKnown mappings:\n{known}")

    if variant != "auto":
        matches = [spec for spec in matches if spec.variant == variant]
    else:
        full_matches = [spec for spec in matches if spec.variant == "full"]
        if full_matches:
            matches = full_matches

    if thinking_mode != "auto":
        matches = [spec for spec in matches if spec.thinking_mode == thinking_mode]

    if len(matches) == 1:
        return matches[0]

    if not matches:
        raise SystemExit(
            "No matching Capability Head after applying --variant/--thinking-mode. "
            "Run with --list to inspect supported choices."
        )

    choices = "\n".join(
        f"  - --model {spec.base_model!r} --variant {spec.variant} "
        f"--thinking-mode {spec.thinking_mode}  -> {spec.repo_id}"
        for spec in matches
    )
    raise SystemExit(f"Ambiguous Capability Head selection. Please disambiguate:\n{choices}")


def safe_path_name(value: str) -> str:
    text = str(value).strip().replace("\\", "/").strip("/")
    text = text.replace("/", "__")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)


def ambiguous_base_variant_pairs() -> set[tuple[str, str]]:
    counts: dict[tuple[str, str], int] = {}
    for spec in CAPABILITY_HEADS:
        key = (safe_path_name(spec.base_model), spec.variant)
        counts[key] = counts.get(key, 0) + 1
    return {key for key, count in counts.items() if count > 1}


def variant_dir_name(spec: CapabilityHeadSpec) -> str:
    key = (safe_path_name(spec.base_model), spec.variant)
    if key in ambiguous_base_variant_pairs():
        return f"{spec.variant}_thinking_{spec.thinking_mode}"
    return spec.variant


def default_output_dir_for_spec(data_root: Path, spec: CapabilityHeadSpec) -> Path:
    return (
        data_root
        / "trained_models"
        / "baseline_capability_heads"
        / safe_path_name(spec.base_model)
        / variant_dir_name(spec)
    )


def iter_download_files(include_readme: bool) -> Iterable[tuple[str, bool]]:
    yield "capability_head.pt", True
    yield "capability_head_config.json", True
    if include_readme:
        yield "README.md", False


def print_known_specs() -> None:
    for spec in CAPABILITY_HEADS:
        print(
            f"{spec.base_model}\tvariant={spec.variant}\tthinking={spec.thinking_mode}\t"
            f"repo={spec.repo_id}\t{spec.description}"
        )


def download_file(
    *,
    repo_id: str,
    filename: str,
    output_dir: Path,
    cache_dir: Path,
    revision: str | None,
    token: str | None,
    force: bool,
    required: bool,
) -> Path | None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit("Missing dependency: pip install huggingface-hub") from exc

    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            revision=revision,
            local_dir=output_dir,
            cache_dir=cache_dir,
            token=token,
            force_download=force,
        )
    except Exception:
        if required:
            raise
        print(f"[skip] optional file not found: {filename}")
        return None
    return Path(path).resolve()


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def download_head_to_dir(
    *,
    repo_id: str,
    spec: CapabilityHeadSpec | None,
    output_dir: Path,
    cache_dir: Path,
    args: argparse.Namespace,
) -> None:
    print(f"[select] repo_id={repo_id}")
    if spec is not None:
        print(f"[select] base_model={spec.base_model} variant={spec.variant} thinking_mode={spec.thinking_mode}")
    print(f"[paths] output_dir={rel(output_dir)}")
    print(f"[paths] hf_cache={rel(cache_dir)}")
    if args.dry_run:
        print("[dry-run] no files downloaded")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded: dict[str, str] = {}
    for filename, required in iter_download_files(include_readme=not args.no_readme):
        path = download_file(
            repo_id=repo_id,
            filename=filename,
            output_dir=output_dir,
            cache_dir=cache_dir,
            revision=args.revision,
            token=args.token,
            force=bool(args.force),
            required=required,
        )
        if path is not None:
            downloaded[filename] = str(path)
            print(f"[download] {filename} -> {rel(path)}")

    config_path = output_dir / "capability_head_config.json"
    config = read_json(config_path) if config_path.exists() else {}
    manifest = {
        "requested_model": args.model,
        "repo_id": repo_id,
        "revision": args.revision,
        "output_dir": str(output_dir),
        "downloaded_files": downloaded,
        "matched_spec": None if spec is None else asdict(spec),
        "capability_head_config": config,
        "source_url": f"https://huggingface.co/{repo_id}",
    }
    manifest_path = output_dir / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[write] {rel(manifest_path)}")
    print(f"[done] capability_head_pt={rel(output_dir / 'capability_head.pt')}")


def main() -> None:
    args = parse_args()
    if args.list:
        print_known_specs()
        return

    if not args.all and not args.model and not args.repo_id:
        raise SystemExit("Pass --all, --model <backbone id/path>, or --repo-id <head repo id>.")
    if args.all and (args.model or args.repo_id):
        raise SystemExit("--all cannot be combined with --model or --repo-id.")

    data_root = resolve_from_code_root(args.data_root)
    ensure_mhlc_data_layout(data_root)
    set_hf_dirs_inside_data_root(data_root)
    cache_dir = data_root / "downloads" / "hf_runtime_cache" / "hub"

    if args.all:
        if args.output_dir:
            root = resolve_from_code_root(args.output_dir)
            output_dirs = {
                spec: root / safe_path_name(spec.base_model) / variant_dir_name(spec)
                for spec in CAPABILITY_HEADS
            }
        else:
            output_dirs = {spec: default_output_dir_for_spec(data_root, spec) for spec in CAPABILITY_HEADS}

        for idx, spec in enumerate(CAPABILITY_HEADS, start=1):
            print(f"\n[all] {idx}/{len(CAPABILITY_HEADS)} {spec.description}")
            download_head_to_dir(
                repo_id=spec.repo_id,
                spec=spec,
                output_dir=output_dirs[spec],
                cache_dir=cache_dir,
                args=args,
            )
        return

    spec = None if args.repo_id else choose_spec(args.model, args.variant, args.thinking_mode)
    repo_id = args.repo_id or spec.repo_id

    if args.output_dir:
        output_dir = resolve_from_code_root(args.output_dir)
    elif spec is not None:
        output_dir = default_output_dir_for_spec(data_root, spec)
    else:
        output_dir = (
            data_root
            / "trained_models"
            / "baseline_capability_heads"
            / "manual_repo"
            / safe_path_name(repo_id)
        )

    download_head_to_dir(
        repo_id=repo_id,
        spec=spec,
        output_dir=output_dir,
        cache_dir=cache_dir,
        args=args,
    )


if __name__ == "__main__":
    main()
