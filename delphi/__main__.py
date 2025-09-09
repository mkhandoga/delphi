import asyncio
import logging
import os
from functools import partial
from pathlib import Path
from typing import Callable

import orjson
import torch
from simple_parsing import ArgumentParser
from torch import Tensor
from transformers import (
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from delphi import logger
from delphi.clients import Offline, OpenRouter
from delphi.config import RunConfig
from delphi.explainers import ContrastiveExplainer, DefaultExplainer, NoOpExplainer
from delphi.explainers.explainer import ExplainerResult
from delphi.latents import LatentCache, LatentDataset
from delphi.latents.neighbours import NeighbourCalculator
from delphi.log.result_analysis import log_results
from delphi.pipeline import Pipe, Pipeline, process_wrapper
from delphi.scorers import DetectionScorer, FuzzingScorer, OpenAISimulator
from delphi.sparse_coders import load_hooks_sparse_coders, load_sparse_coders
from delphi.utils import assert_type, load_tokenized_data

import re
from typing import Optional, Dict, List

def build_latent_selection_from_graph(
    graph_path: Path,
    allowed_modules: Optional[List[str]] = None,
    latents_root: Optional[Path] = None,
    limit_per_module: Optional[int] = None,   # <-- count-based limit
) -> Dict[str, torch.Tensor]:
    """
    Parse a circuit graph JSON and return:
        { module (e.g., 'layers.11.mlp'): tensor([latent_ids...], dtype=long) }

    Recognizes node_id like: 'intermediate_<run>_<layer>_<latent>'.
    If `allowed_modules` is provided, keep only those.
    If `latents_root` is provided, clamp indices to what exists on disk (via firing_counts.pt).
    If `limit_per_module` is provided, keep at most that many ids per module (by sorted id).
    """
    import re, orjson, torch
    buf: Dict[str, set] = {}
    data = orjson.loads(graph_path.read_bytes())

    nodes = data.get("nodes", [])
    pat = re.compile(r"^intermediate_(\d+)_([\d]+)_([\d]+)$")

    for n in nodes:
        node_id = n.get("node_id") or ""
        m = pat.match(node_id)
        if not m:
            continue
        layer = int(m.group(2))
        latent_idx = int(m.group(3))
        module = f"layers.{layer}.mlp"
        if allowed_modules is not None and module not in allowed_modules:
            continue
        buf.setdefault(module, set()).add(latent_idx)

    out: Dict[str, torch.Tensor] = {}
    for module, idxs in buf.items():
        idx_list = sorted(idxs)

        # optionally clamp to available latents on disk
        if latents_root is not None:
            per_mod_dir = latents_root / module
            fc_file = per_mod_dir / "firing_counts.pt"
            if fc_file.exists():
                counts = torch.load(fc_file, weights_only=True)
                max_ok = counts.numel()
                idx_list = [i for i in idx_list if i < max_ok]

        # limit by COUNT, not by VALUE
        if limit_per_module is not None:
            idx_list = idx_list[:limit_per_module]

        if idx_list:
            out[module] = torch.tensor(idx_list, dtype=torch.long)

    return out


def load_artifacts(run_cfg: RunConfig):
    if run_cfg.load_in_8bit:
        dtype = torch.float16
    elif torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = "auto"

    # Don't override quantization_config for pre-quantized models
    model_kwargs = {
        "device_map": "auto",  # Let it distribute across available GPUs
        "torch_dtype": dtype,
        "token": run_cfg.hf_token,
    }
    
    # Only add quantization_config if explicitly using 8-bit
    if run_cfg.load_in_8bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    
    model = AutoModel.from_pretrained(run_cfg.model, **model_kwargs)

    hookpoint_to_sparse_encode, transcode = load_hooks_sparse_coders(
        model,
        run_cfg,
        compile=True,
    )

    return (
        list(hookpoint_to_sparse_encode.keys()),
        hookpoint_to_sparse_encode,
        model,
        transcode,
    )


def create_neighbours(
    run_cfg: RunConfig,
    latents_path: Path,
    neighbours_path: Path,
    hookpoints: list[str],
):
    """
    Creates a neighbours file for the given hookpoints.
    """
    neighbours_path.mkdir(parents=True, exist_ok=True)

    constructor_cfg = run_cfg.constructor_cfg
    saes = (
        load_sparse_coders(run_cfg, device="cpu")
        if constructor_cfg.neighbours_type != "co-occurrence"
        else {}
    )

    for hookpoint in hookpoints:

        if constructor_cfg.neighbours_type == "co-occurrence":
            neighbour_calculator = NeighbourCalculator(
                cache_dir=latents_path / hookpoint, number_of_neighbours=250
            )

        elif constructor_cfg.neighbours_type == "decoder_similarity":

            neighbour_calculator = NeighbourCalculator(
                autoencoder=saes[hookpoint].to("cuda"), number_of_neighbours=250
            )

        elif constructor_cfg.neighbours_type == "encoder_similarity":
            neighbour_calculator = NeighbourCalculator(
                autoencoder=saes[hookpoint].to("cuda"), number_of_neighbours=250
            )
        else:
            raise ValueError(
                f"Neighbour type {constructor_cfg.neighbours_type} not supported"
            )

        neighbour_calculator.populate_neighbour_cache(constructor_cfg.neighbours_type)
        neighbour_calculator.save_neighbour_cache(f"{neighbours_path}/{hookpoint}")


async def process_cache(
    run_cfg: RunConfig,
    latents_path: Path,
    explanations_path: Path,
    scores_path: Path,
    hookpoints: list[str],
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    latent_range: Tensor | None,
    latent_selection: Optional[Dict[str, torch.Tensor]] = None,  # <-- NEW
):
    """
    Converts SAE latent activations in on-disk cache in the `latents_path` directory
    to latent explanations in the `explanations_path` directory and explanation
    scores in the `scores_path` directory.
    """
    explanations_path.mkdir(parents=True, exist_ok=True)
    
    if latent_selection is not None:
        # from circuit graph
        latent_dict = latent_selection
    elif latent_range is not None:
        # from --max_latents
        latent_dict = {hook: latent_range for hook in hookpoints}
    else:
        # use all available latents for each module
        latent_dict = None

    dataset = LatentDataset(
        raw_dir=latents_path,
        sampler_cfg=run_cfg.sampler_cfg,
        constructor_cfg=run_cfg.constructor_cfg,
        modules=hookpoints,
        latents=latent_dict,
        tokenizer=tokenizer,
    )

    if run_cfg.explainer_provider == "offline":
        llm_client = Offline(
            run_cfg.explainer_model,
            max_memory=0.7,
            # Explainer models context length - must be able to accommodate the longest
            # set of examples
            max_model_len=run_cfg.explainer_model_max_len,
            num_gpus=run_cfg.num_gpus,
            statistics=run_cfg.verbose,
        )
    elif run_cfg.explainer_provider == "openrouter":
        if (
            "OPENROUTER_API_KEY" not in os.environ
            or not os.environ["OPENROUTER_API_KEY"]
        ):
            raise ValueError(
                "OPENROUTER_API_KEY environment variable not set. Set "
                "`--explainer-provider offline` to use a local explainer model."
            )

        llm_client = OpenRouter(
            run_cfg.explainer_model,
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
    else:
        raise ValueError(
            f"Explainer provider {run_cfg.explainer_provider} not supported"
        )

    if not run_cfg.explainer == "none":

        def explainer_postprocess(result):
            with open(explanations_path / f"{result.record.latent}.txt", "wb") as f:
                f.write(orjson.dumps(result.explanation))

            return result

        if run_cfg.constructor_cfg.non_activating_source == "FAISS":
            explainer = ContrastiveExplainer(
                llm_client,
                threshold=0.3,
                verbose=run_cfg.verbose,
            )
        else:
            explainer = DefaultExplainer(
                llm_client,
                threshold=0.3,
                verbose=run_cfg.verbose,
            )

        explainer_pipe = Pipe(
            process_wrapper(explainer, postprocess=explainer_postprocess)
        )
    else:

        def none_postprocessor(result):
            # Load the explanation from disk
            explanation_path = explanations_path / f"{result.record.latent}.txt"
            if not explanation_path.exists():
                raise FileNotFoundError(
                    f"Explanation file {explanation_path} does not exist. "
                    "Make sure to run an explainer pipeline first."
                )

            with open(explanation_path, "rb") as f:
                return ExplainerResult(
                    record=result.record,
                    explanation=orjson.loads(f.read()),
                )

        explainer_pipe = Pipe(
            process_wrapper(
                NoOpExplainer(),
                postprocess=none_postprocessor,
            )
        )

    # Builds the record from result returned by the pipeline
    def scorer_preprocess(result):
        if isinstance(result, list):
            result = result[0]

        record = result.record
        record.explanation = result.explanation
        record.extra_examples = record.not_active
        return record

    # Saves the score to a file
    def scorer_postprocess(result, score_dir):
        safe_latent_name = str(result.record.latent).replace("/", "--")

        with open(score_dir / f"{safe_latent_name}.txt", "wb") as f:
            f.write(orjson.dumps(result.score))

    scorers = []
    for scorer_name in run_cfg.scorers:
        scorer_path = scores_path / scorer_name
        scorer_path.mkdir(parents=True, exist_ok=True)

        if scorer_name == "simulation":
            scorer = OpenAISimulator(llm_client, tokenizer=tokenizer, all_at_once=False)
        elif scorer_name == "fuzz":
            scorer = FuzzingScorer(
                llm_client,
                n_examples_shown=run_cfg.num_examples_per_scorer_prompt,
                verbose=run_cfg.verbose,
                log_prob=run_cfg.log_probs,
            )
        elif scorer_name == "detection":
            scorer = DetectionScorer(
                llm_client,
                n_examples_shown=run_cfg.num_examples_per_scorer_prompt,
                verbose=run_cfg.verbose,
                log_prob=run_cfg.log_probs,
            )
        else:
            raise ValueError(f"Scorer {scorer_name} not supported")

        wrapped_scorer = process_wrapper(
            scorer,
            preprocess=scorer_preprocess,
            postprocess=partial(scorer_postprocess, score_dir=scorer_path),
        )
        scorers.append(wrapped_scorer)

    pipeline = Pipeline(
        dataset,
        explainer_pipe,
        Pipe(*scorers),
    )

    if run_cfg.pipeline_num_proc > 1 and run_cfg.explainer_provider == "openrouter":
        print(
            "OpenRouter does not support multiprocessing,"
            " setting pipeline_num_proc to 1"
        )
        run_cfg.pipeline_num_proc = 1

    await pipeline.run(run_cfg.pipeline_num_proc)


def populate_cache(
    run_cfg: RunConfig,
    model: PreTrainedModel,
    hookpoint_to_sparse_encode: dict[str, Callable],
    latents_path: Path,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    transcode: bool,
):
    """
    Populates an on-disk cache in `latents_path` with SAE latent activations.
    """
    latents_path.mkdir(parents=True, exist_ok=True)

    # Create a log path within the run directory
    log_path = latents_path.parent / "log"
    log_path.mkdir(parents=True, exist_ok=True)

    cache_cfg = run_cfg.cache_cfg
    tokens = load_tokenized_data(
        cache_cfg.cache_ctx_len,
        tokenizer,
        cache_cfg.dataset_repo,
        cache_cfg.dataset_split,
        cache_cfg.dataset_name,
        cache_cfg.dataset_column,
        run_cfg.seed,
    )

    if run_cfg.filter_bos:
        if tokenizer.bos_token_id is None:
            print("Tokenizer does not have a BOS token, skipping BOS filtering")
        else:
            flattened_tokens = tokens.flatten()
            mask = ~torch.isin(flattened_tokens, torch.tensor([tokenizer.bos_token_id]))
            masked_tokens = flattened_tokens[mask]
            truncated_tokens = masked_tokens[
                : len(masked_tokens) - (len(masked_tokens) % cache_cfg.cache_ctx_len)
            ]
            tokens = truncated_tokens.reshape(-1, cache_cfg.cache_ctx_len)

    cache = LatentCache(
        model,
        hookpoint_to_sparse_encode,
        batch_size=cache_cfg.batch_size,
        transcode=transcode,
        log_path=log_path,
    )
    cache.run(cache_cfg.n_tokens, tokens)

    if run_cfg.verbose:
        cache.generate_statistics_cache()

    cache.save_splits(
        # Split the activation and location indices into different files to make
        # loading faster
        n_splits=cache_cfg.n_splits,
        save_dir=latents_path,
    )

    cache.save_config(save_dir=latents_path, cfg=cache_cfg, model_name=run_cfg.model)


def non_redundant_hookpoints(
    hookpoint_to_sparse_encode: dict[str, Callable] | list[str],
    results_path: Path,
    overwrite: bool,
) -> dict[str, Callable] | list[str]:
    """
    Returns a list of hookpoints that are not already in the cache.
    """
    if overwrite:
        print("Overwriting results from", results_path)
        return hookpoint_to_sparse_encode
    in_results_path = [x.name for x in results_path.glob("*")]
    if isinstance(hookpoint_to_sparse_encode, dict):
        non_redundant_hookpoints = {
            k: v
            for k, v in hookpoint_to_sparse_encode.items()
            if k not in in_results_path
        }
    else:
        non_redundant_hookpoints = [
            hookpoint
            for hookpoint in hookpoint_to_sparse_encode
            if hookpoint not in in_results_path
        ]
    if not non_redundant_hookpoints:
        print(f"Files found in {results_path}, skipping...")
    return non_redundant_hookpoints

async def run(run_cfg: RunConfig, graph_path: Optional[Path] = None):
    base_path = Path.cwd() / "results"
    if run_cfg.name:
        base_path = base_path / run_cfg.name
    base_path.mkdir(parents=True, exist_ok=True)

    run_cfg.save_json(base_path / "run_config.json", indent=4)

    latents_path = base_path / "latents"
    explanations_path = base_path / "explanations"
    scores_path = base_path / "scores"
    neighbours_path = base_path / "neighbours"
    visualize_path = base_path / "visualize"

    latent_range = torch.arange(run_cfg.max_latents) if run_cfg.max_latents else None

    # --- Parse the graph FIRST so we know which SAEs to load
    initial_graph_sel: Optional[Dict[str, torch.Tensor]] = None
    if graph_path is not None:
        limit = int(run_cfg.max_latents) if run_cfg.max_latents else None
        initial_graph_sel = build_latent_selection_from_graph(
            graph_path=graph_path,
            allowed_modules=(run_cfg.hookpoints or None),
            latents_root=None,               # clamp later
            limit_per_module=limit,          # <--- count limit
        )
        if initial_graph_sel:
            run_cfg.hookpoints = (
                [m for m in run_cfg.hookpoints if m in initial_graph_sel]
                if run_cfg.hookpoints else sorted(initial_graph_sel.keys())
            )
        else:
            print(f"[graph] No valid latent nodes found in {graph_path}; "
                  f"continuing with --hookpoints/--max_latents.")

    # Load model + SAEs (now that run_cfg.hookpoints reflects the graph)
    hookpoints, hookpoint_to_sparse_encode, model, transcode = load_artifacts(run_cfg)
    tokenizer = AutoTokenizer.from_pretrained(run_cfg.model, token=run_cfg.hf_token)

    # If graph mentioned modules for which no SAE loaded, drop them with a notice
    if initial_graph_sel is not None:
        available = set(hookpoints)
        missing = [m for m in initial_graph_sel if m not in available]
        if missing:
            print(f"[graph] Skipping modules with no SAE: {missing}")
        initial_graph_sel = {m: t for m, t in initial_graph_sel.items() if m in available}
        if initial_graph_sel:
            hookpoints = [m for m in hookpoints if m in initial_graph_sel]
        else:
            print("[graph] After filtering to available SAEs, no modules remain.")

    # Populate cache only for target modules
    encoders_for_targets = {k: v for k, v in hookpoint_to_sparse_encode.items() if k in set(hookpoints)}
    nrh = assert_type(
        dict,
        non_redundant_hookpoints(encoders_for_targets, latents_path, "cache" in run_cfg.overwrite),
    )
    if nrh:
        populate_cache(run_cfg, model, nrh, latents_path, tokenizer, transcode)

    # Free big objects before scoring
    del model, hookpoint_to_sparse_encode

    # Optional neighbours
    if run_cfg.constructor_cfg.non_activating_source == "neighbours":
        nrh = assert_type(
            list,
            non_redundant_hookpoints(hookpoints, neighbours_path, "neighbours" in run_cfg.overwrite),
        )
        if nrh:
            create_neighbours(run_cfg, latents_path, neighbours_path, nrh)
    else:
        print("Skipping neighbour creation")

    # Build FINAL graph-based latent selection, now clamped against what exists on disk
    graph_latent_selection: Optional[Dict[str, torch.Tensor]] = None
    if graph_path is not None:
        limit = int(run_cfg.max_latents) if run_cfg.max_latents else None
        graph_latent_selection = build_latent_selection_from_graph(
            graph_path=graph_path,
            allowed_modules=hookpoints,
            latents_root=latents_path,       # clamp to what's on disk
            limit_per_module=limit,          # <--- count limit
        )

        if not graph_latent_selection:
            print("[graph] No usable latent indices after clamping to cache; "
                  "falling back to --max_latents/ALL for selected modules.")
            graph_latent_selection = None  # fall back to latent_range logic

    target_modules = list(graph_latent_selection.keys()) if graph_latent_selection else hookpoints
    nrh = assert_type(
        list,
        non_redundant_hookpoints(target_modules, scores_path, "scores" in run_cfg.overwrite),
    )
    if nrh:
        await process_cache(
            run_cfg,
            latents_path,
            explanations_path,
            scores_path,
            nrh,
            tokenizer,
            latent_range=None if graph_latent_selection is not None else latent_range,
            latent_selection=graph_latent_selection,
        )

    if run_cfg.verbose:
        processed_modules = list(graph_latent_selection.keys()) if graph_latent_selection else hookpoints
        log_results(scores_path, visualize_path, processed_modules, run_cfg.scorers)


if __name__ == "__main__":
    # Configure logging for CLI usage
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler("delphi.log")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    parser = ArgumentParser()
    parser.add_arguments(RunConfig, dest="run_cfg")
    parser.add_argument(
        "--graph",
        type=Path,
        default=None,
        help="Path to a circuit graph JSON. If set, Delphi will explain only the "
             "features referenced in the graph. Overrides --max_latents selection.",
    )
    args = parser.parse_args()

    asyncio.run(run(args.run_cfg, graph_path=args.graph))   # ✅