from safetensors.torch import load_file
from pathlib import Path

shard = Path("../attribute/g2_trans_cache/latents/layers.11.mlp/0_3275.safetensors")
sd = load_file(str(shard))
print("Keys:", list(sd.keys()))
for k, v in sd.items():
    print(k, v.shape, v.dtype)
