from pathlib import Path
import torch, sys

run_dir = Path("results/my_run")
mod = "layers.11.mlp"
per_mod = run_dir / "latents" / mod / "firing_counts.pt"
agg = run_dir / "log" / "hookpoint_firing_counts.pt"
agg.parent.mkdir(parents=True, exist_ok=True)

counts = torch.load(per_mod, weights_only=True)
torch.save({mod: counts}, agg)
print(f"wrote {agg} with key '{mod}' and shape {tuple(counts.shape)}")