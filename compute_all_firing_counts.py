#!/usr/bin/env python3
"""
compute_all_firing_counts.py

Compute per-latent firing counts + token totals for every hookpoint under a latents/ dir.

Outputs per run:
  • <run>/log/hookpoint_firing_counts.pt        # dict[module] = LongTensor[num_latents]
  • <run>/log/hookpoint_token_counts.json       # dict[module] = int (approx total tokens)
  • <run>/latents/<module>/firing_counts.pt     # LongTensor[num_latents]
  • <run>/latents/<module>/firing_counts.json   # list[int] (for quick inspection)
  • <run>/latents/<module>/token_count.json     # {"num_tokens": int}

Notes:
- Latent index is assumed to be in locations[:, 2]. Override with --latent-col if needed.
- If per-module files already exist, use --overwrite to recompute them.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import orjson
import torch
from safetensors.torch import load_file


def list_module_dirs(latents_root: Path) -> List[Path]:
    return sorted([p for p in latents_root.iterdir() if p.is_dir()])


def list_shards(module_dir: Path) -> List[Path]:
    return sorted([p for p in module_dir.glob("*.safetensors") if p.is_file()])


def bincount_extend(ids: torch.Tensor, out: torch.Tensor | None) -> torch.Tensor:
    """Accumulate bincount of `ids` into `out`, growing it as needed."""
    if ids.numel() == 0:
        return out if out is not None else torch.zeros(0, dtype=torch.long)
    ids = ids.to(torch.int64).view(-1).cpu()
    needed = int(ids.max().item()) + 1
    if out is None:
        return torch.bincount(ids, minlength=needed)
    if needed > out.numel():
        out = torch.nn.functional.pad(out, (0, needed - out.numel()))
    out += torch.bincount(ids, minlength=out.numel())
    return out


def compute_counts_and_tokens_for_module(
    module_dir: Path, latent_col: int = 2
) -> Tuple[torch.Tensor, int]:
    """
    Returns:
        counts: LongTensor[num_latents]
        total_tokens: int (approx; sum over shards of num_seqs * seq_len)
    """
    counts = None
    total_tokens = 0
    shards = list_shards(module_dir)
    if not shards:
        raise FileNotFoundError(f"No .safetensors shards found in {module_dir}")

    for f in shards:
        sd = load_file(str(f))  # CPU tensors
        if "locations" not in sd:
            raise KeyError(f"'locations' not found in {f.name} (keys: {list(sd.keys())})")
        loc = sd["locations"]  # [N, 3] -> (seq_id, pos, latent_idx)
        if loc.dim() != 2 or loc.size(1) <= latent_col:
            raise ValueError(f"Unexpected 'locations' shape {tuple(loc.shape)} in {f.name}")
        latent_ids = loc[:, latent_col]
        counts = bincount_extend(latent_ids, counts)

        # Optional tokens estimate (if present): [num_seqs, seq_len]
        if "tokens" in sd:
            t = sd["tokens"]
            if t.dim() == 2:
                total_tokens += int(t.size(0) * t.size(1))

        del sd, loc, latent_ids

    if counts is None:
        counts = torch.zeros(0, dtype=torch.long)
    return counts.to(torch.long), total_tokens


def main():
    ap = argparse.ArgumentParser(description="Compute firing counts & token totals for all modules in a latents/ dir.")
    ap.add_argument("--latents-root", type=Path, required=True,
                    help="Path to results/<run>/latents")
    ap.add_argument("--latent-col", type=int, default=2,
                    help="Column in 'locations' with latent indices (default: 2)")
    ap.add_argument("--write-json", action="store_true",
                    help="Also write per-module firing_counts.json")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing per-module outputs and aggregate files")
    args = ap.parse_args()

    latents_root = args.latents_root.resolve()
    if not latents_root.exists() or not latents_root.is_dir():
        raise SystemExit(f"Not a directory: {latents_root}")

    # Infer run dir and aggregate output paths
    run_dir = latents_root.parent
    log_dir = run_dir / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    agg_counts_out = log_dir / "hookpoint_firing_counts.pt"
    agg_tokens_out = log_dir / "hookpoint_token_counts.json"

    counts_map: Dict[str, torch.Tensor] = {}
    token_map: Dict[str, int] = {}

    total_modules = 0
    grand_tokens = 0

    for module_dir in list_module_dirs(latents_root):
        module = module_dir.name
        # Skip folders without shards (e.g., stray files or config dirs)
        if not list_shards(module_dir):
            continue

        per_module_pt = module_dir / "firing_counts.pt"
        per_module_json = module_dir / "firing_counts.json"
        per_module_tokens_json = module_dir / "token_count.json"

        # Decide whether to recompute
        need_recompute = args.overwrite or not per_module_pt.exists()

        if need_recompute:
            counts, num_tokens = compute_counts_and_tokens_for_module(module_dir, latent_col=args.latent_col)
            torch.save(counts, per_module_pt)
            if args.write_json:
                per_module_json.write_bytes(orjson.dumps(counts.tolist()))
            per_module_tokens_json.write_bytes(orjson.dumps({"num_tokens": num_tokens}))
        else:
            counts = torch.load(per_module_pt, weights_only=True)
            # Load num_tokens if present; otherwise compute just tokens quickly
            if per_module_tokens_json.exists():
                num_tokens = orjson.loads(per_module_tokens_json.read_bytes()).get("num_tokens", 0)
            else:
                # Quick token-only pass
                num_tokens = 0
                for f in list_shards(module_dir):
                    sd = load_file(str(f))
                    if "tokens" in sd and sd["tokens"].dim() == 2:
                        t = sd["tokens"]
                        num_tokens += int(t.size(0) * t.size(1))
                per_module_tokens_json.write_bytes(orjson.dumps({"num_tokens": num_tokens}))

        counts_map[module] = counts
        token_map[module] = int(num_tokens)
        grand_tokens += int(num_tokens)

        nonzero = int((counts > 0).sum().item())
        print(f"[ok] {module:20s} latents={counts.numel():6d} active={nonzero:6d} "
              f"firings={int(counts.sum().item()):10d} tokens~{num_tokens:,}")
        total_modules += 1

    if not counts_map:
        raise SystemExit("No modules processed. Did you point --latents-root to the correct directory?")

    # Write aggregates
    if agg_counts_out.exists() and not args.overwrite:
        print(f"[skip] {agg_counts_out} exists (use --overwrite to rewrite)")
    else:
        torch.save(counts_map, agg_counts_out)
        print(f"[done] Wrote aggregate counts → {agg_counts_out}")

    if agg_tokens_out.exists() and not args.overwrite:
        print(f"[skip] {agg_tokens_out} exists (use --overwrite to rewrite)")
    else:
        agg_tokens_out.write_bytes(orjson.dumps(token_map))
        print(f"[done] Wrote aggregate token totals → {agg_tokens_out}")

    print(f"\nSummary: modules={total_modules}   approx tokens (sum)={grand_tokens:,}")


if __name__ == "__main__":
    main()
