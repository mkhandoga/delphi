from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Optional, Protocol, Union

import torch
from sparsify import SparseCoder, SparseCoderConfig
from sparsify.sparse_coder import EncoderOutput
from torch import Tensor
from transformers import PreTrainedModel


class PotentiallyWrappedSparseCoder(Protocol):
    def encode(self, x: Tensor) -> EncoderOutput: ...

    def to(
        self,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.nn.Module: ...

    cfg: SparseCoderConfig
    num_latents: int

from safetensors import safe_open
from safetensors.torch import load_file, save_file

def _strip_unexpected_keys_in_dir(dir_path: Path, bad_prefixes=("post_enc",)) -> bool:
    """
    Remove parameters whose names start with any of bad_prefixes from all
    .safetensors files in dir_path. Returns True if anything was changed.
    """
    changed = False
    for f in dir_path.glob("*.safetensors"):
        # Fast peek at keys without loading tensors to GPU
        with safe_open(f, framework="pt") as sf:
            keys = list(sf.keys())
        drop = [k for k in keys if any(k == p or k.startswith(p + ".") for p in bad_prefixes)]
        if not drop:
            continue

        sd = load_file(str(f))  # tensors are still on CPU here
        keep = {k: v for k, v in sd.items() if k not in drop}
        backup = f.with_suffix(f.suffix + ".bak")
        f.rename(backup)
        save_file(keep, str(f))
        changed = True
        print(f"[delphi] Stripped {len(drop)} unexpected keys from {f.name}: {drop[:5]}{' ...' if len(drop) > 5 else ''}")
    return changed

# def sae_dense_latents(x: Tensor, sae: PotentiallyWrappedSparseCoder) -> Tensor:
#     """Run `sae` on `x`, yielding the dense activations."""
#     x_in = x.reshape(-1, x.shape[-1])
#     encoded = sae.encode(x_in)
#     buf = torch.zeros(
#         x_in.shape[0], sae.num_latents, dtype=x_in.dtype, device=x_in.device
#     )
#     buf = buf.scatter_(-1, encoded.top_indices, encoded.top_acts.to(buf.dtype))
#     return buf.reshape(*x.shape[:-1], -1)

def sae_dense_latents(x: Tensor, sae: PotentiallyWrappedSparseCoder) -> Tensor:
    """Run `sae` on `x`, yielding dense activations.
    Cast input to the SAE's parameter dtype/device to avoid dtype mismatches.
    """
    # Flatten tokens×positions
    x_in = x.reshape(-1, x.shape[-1])

    # Discover SAE dtype/device
    try:
        p = next(sae.parameters())
        sae_device = p.device
        sae_dtype = p.dtype
    except Exception:
        sae_device = x_in.device
        sae_dtype = x_in.dtype

    # Move & cast input to match SAE weights (and keep it contiguous for fused kernels)
    x_in = x_in.to(device=sae_device, dtype=sae_dtype).contiguous()

    # Encode (topk indices + activations)
    encoded = sae.encode(x_in)

    # Build dense output in SAE compute dtype, then return in the original x dtype
    buf = torch.zeros(
        x_in.shape[0], sae.num_latents, dtype=sae_dtype, device=sae_device
    )
    buf.scatter_(-1, encoded.top_indices, encoded.top_acts.to(buf.dtype))

    return buf.reshape(*x.shape[:-1], -1).to(dtype=x.dtype, device=x.device)


def resolve_path(
    model: PreTrainedModel | torch.nn.Module, path_segments: list[str]
) -> list[str] | None:
    """Attempt to resolve the path segments to the model in the case where it
    has been wrapped (e.g. by a LanguageModel, causal model, or classifier)."""
    # If the first segment is a valid attribute, return the path segments
    if hasattr(model, path_segments[0]):
        return path_segments

    # Look for the first actual model inside potential wrappers
    for attr_name, attr in model.named_children():
        if isinstance(attr, torch.nn.Module):
            print(f"Checking wrapper model attribute: {attr_name}")
            if hasattr(attr, path_segments[0]):
                print(
                    f"Found matching path in wrapper at {attr_name}.{path_segments[0]}"
                )
                return [attr_name] + path_segments

            # Recursively check deeper
            deeper_path = resolve_path(attr, path_segments)
            if deeper_path is not None:
                print(f"Found deeper matching path starting with {attr_name}")
                return [attr_name] + deeper_path
    return None


def load_sparsify_sparse_coders(
    name: str,
    hookpoints: list[str],
    device: str | torch.device,
    compile: bool = False,
) -> dict[str, PotentiallyWrappedSparseCoder]:
    sparse_model_dict = {}
    name_path = Path(name)

    if name_path.exists():
        for hookpoint in hookpoints:
            hook_dir = name_path / hookpoint

            # NEW: sanitize unexpected keys (e.g., 'post_enc') so strict load won't fail
            _strip_unexpected_keys_in_dir(hook_dir, bad_prefixes=("post_enc",))

            try:
                sc = SparseCoder.load_from_disk(hook_dir, device=device)
            except RuntimeError as e:
                # Helpful message if something new pops up
                raise RuntimeError(
                    f"SparseCoder.load_from_disk failed for {hook_dir}. "
                    f"If this is an 'Unexpected key(s)...' error, your checkpoint "
                    f"contains parameters not supported by the current 'sparsify'. "
                    f"Either re-export without those modules or update 'sparsify'.\n\n{e}"
                ) from e

            if compile:
                sc = torch.compile(sc)
            sparse_model_dict[hookpoint] = sc

    else:
        # remote/hub path
        sparse_models = SparseCoder.load_many(name, device="cpu")
        for hookpoint in hookpoints:
            sc = sparse_models[hookpoint].to(device)
            if compile:
                sc = torch.compile(sc)
            sparse_model_dict[hookpoint] = sc
        del sparse_models

    return sparse_model_dict


def load_sparsify_hooks(
    model: PreTrainedModel,
    name: str,
    hookpoints: list[str],
    device: str | torch.device | None = None,
    compile: bool = False,
) -> tuple[dict[str, Callable], bool]:
    """
    Load the encode functions for sparsify sparse coders on specified hookpoints.

    Args:
        model (Any): The model to load autoencoders for.
        name (str): The name of the sparse model to load. If the model is on-disk
            this is the path to the directory containing the sparse model weights.
        hookpoints (list[str]): list of hookpoints to identify the sparse models.
        device (str | torch.device | None, optional): The device to load the
            sparse models on. If not specified the sparse models will be loaded
            on the same device as the base model.

    Returns:
        dict[str, Callable]: A dictionary mapping hookpoints to encode functions.
    """
    device = model.device or "cpu"
    sparse_model_dict = load_sparsify_sparse_coders(
        name,
        hookpoints,
        device,
        compile,
    )
    hookpoint_to_sparse_encode = {}
    transcode = False
    for hookpoint, sparse_model in sparse_model_dict.items():
        print(f"Resolving path for hookpoint: {hookpoint}")
        path_segments = resolve_path(model, hookpoint.split("."))
        if path_segments is None:
            raise ValueError(f"Could not find valid path for hookpoint: {hookpoint}")

        hookpoint_to_sparse_encode[".".join(path_segments)] = partial(
            sae_dense_latents, sae=sparse_model
        )
        # We only need to check if one of the sparse models is a transcoder
        if hasattr(sparse_model.cfg, "transcode"):
            if sparse_model.cfg.transcode:
                transcode = True
        if hasattr(sparse_model.cfg, "skip_connection"):
            if sparse_model.cfg.skip_connection:
                transcode = True
    return hookpoint_to_sparse_encode, transcode
