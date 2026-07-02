"""Gene panel selection (CuPy HVG) and pretrained model weight slicing."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


def _cuda_device_id() -> int:
    raw = os.environ.get("HVG_CUDA_DEVICE", os.environ.get("CUDA_DEVICE", "")).strip()
    import cupy as cp

    if int(cp.cuda.runtime.getDeviceCount()) <= 0:
        raise RuntimeError("CUDA required for HVG (CuPy GPU block accumulation)")
    return int(raw) if raw else 0


def _var_column_index(backed: ad.AnnData, var_names: pd.Index | None) -> tuple[np.ndarray | None, pd.Index]:
    if var_names is None:
        return None, pd.Index(map(str, backed.var_names))
    want = pd.Index(map(str, var_names))
    col_idx = np.asarray(backed.var_names.get_indexer(want), dtype=np.int64)
    if (col_idx < 0).any():
        missing = want[col_idx < 0][:5].tolist()
        raise ValueError(f"HVG column slice: {len(want[col_idx < 0])} genes missing (e.g. {missing})")
    return col_idx, want


def _accumulate_block_gpu(x, total, total_sq, counts, *, dense_rows: int, cp, cpx_sp) -> None:
    if sp.issparse(x):
        counts += cp.asarray(x.getnnz(axis=0), dtype=cp.float64)
        gx = cpx_sp.csr_matrix(x)
        total += gx.sum(axis=0).ravel()
        total_sq += gx.power(2).sum(axis=0).ravel()
    else:
        gx = cp.asarray(x, dtype=cp.float64)
        total += gx.sum(axis=0)
        total_sq += cp.square(gx).sum(axis=0)
        counts += cp.float64(dense_rows)


def gpu_stream_hvg_stats(
    h5ad_path: str | Path,
    *,
    block_rows: int | None = None,
    desc: str | None = None,
    var_names: pd.Index | list[str] | None = None,
    device_id: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Index]:
    """Accumulate per-gene mean/var on GPU from backed HDF5 row blocks."""
    import cupy as cp
    import cupyx.scipy.sparse as cpx_sp

    device_id = _cuda_device_id() if device_id is None else device_id
    block_rows = block_rows or _env_int("HVG_GPU_BLOCK_ROWS", 50_000)
    backed = ad.read_h5ad(h5ad_path, backed="r")
    try:
        col_idx, out_var = _var_column_index(
            backed, pd.Index(map(str, var_names)) if var_names is not None else None
        )
        n_obs, n_vars = int(backed.n_obs), len(out_var)
        label = desc or Path(h5ad_path).name
        with cp.cuda.Device(device_id):
            total = cp.zeros(n_vars, dtype=cp.float64)
            total_sq = cp.zeros(n_vars, dtype=cp.float64)
            counts = cp.zeros(n_vars, dtype=cp.float64)
            with tqdm(total=n_obs, desc=f"{label} (gpu)", unit="cell", mininterval=1.0) as bar:
                for start in range(0, n_obs, block_rows):
                    end = min(start + block_rows, n_obs)
                    x = backed[start:end, col_idx].X if col_idx is not None else backed[start:end, :].X
                    _accumulate_block_gpu(
                        x, total, total_sq, counts, dense_rows=end - start, cp=cp, cpx_sp=cpx_sp
                    )
                    bar.update(end - start)
            counts = cp.maximum(counts, 1.0)
            mean = total / counts
            var = cp.maximum(total_sq / counts - mean * mean, 0.0)
            return cp.asnumpy(mean), cp.asnumpy(var), cp.asnumpy(counts), out_var
    finally:
        if getattr(backed, "file", None) is not None:
            backed.file.close()


def select_hvg(
    mean: np.ndarray,
    var: np.ndarray,
    var_names: pd.Index,
    *,
    n_top_genes: int | None = None,
    min_mean: float = 0.0125,
    max_mean: float = 3.0,
    min_disp: float = 0.5,
) -> pd.Index:
    """Dispersion-based HVG selection from streaming stats."""
    n_top_genes = n_top_genes or _env_int("HVG_N_TOP", 4000)
    mean = np.asarray(mean, dtype=np.float64)
    var = np.asarray(var, dtype=np.float64)
    mask = (mean >= min_mean) & (mean <= max_mean) & (var > 0)
    if not bool(mask.any()):
        order = np.argsort(-var)
        return var_names[order[: min(n_top_genes, len(var_names))]]
    disp = var[mask] / (mean[mask] + 1e-12)
    within = var_names[mask]
    order = np.argsort(-disp)
    picked = within[order[: min(n_top_genes, len(within))]]
    return pd.Index(picked)


def gpu_hvg_genes(h5ad_path: str | Path, **kwargs) -> pd.Index:
    stats_kw = {
        k: kwargs[k]
        for k in ("block_rows", "desc", "var_names", "device_id")
        if k in kwargs
    }
    select_kw = {k: kwargs[k] for k in ("n_top_genes", "min_mean", "max_mean", "min_disp") if k in kwargs}
    mean, var, _counts, var_names = gpu_stream_hvg_stats(h5ad_path, **stats_kw)
    return select_hvg(mean, var, var_names, **select_kw)


def resolve_gene_panel(
    model_var_names: pd.Index | list[str],
    hvg_obs: pd.Index,
    hvg_parse: pd.Index,
) -> pd.Index:
    """model_var ∩ hvg_obs ∩ hvg_parse in model var order."""
    model = pd.Index(map(str, model_var_names))
    obs_set = set(map(str, hvg_obs))
    parse_set = set(map(str, hvg_parse))
    return model[model.isin(obs_set) & model.isin(parse_set)]


def resolve_parse_gene_panel(
    obs_path: str | Path,
    parse_path: str | Path,
    model_var_names: pd.Index,
    *,
    parse_var_slice: pd.Index | None = None,
) -> pd.Index:
    obs_path, parse_path = Path(obs_path), Path(parse_path)
    logger.info("GPU HVG: observational %s", obs_path)
    hvg_obs = gpu_hvg_genes(obs_path, desc="HVG observational")
    logger.info("GPU HVG: Parse %s", parse_path)
    hvg_parse = gpu_hvg_genes(parse_path, var_names=parse_var_slice, desc="HVG Parse")
    panel = resolve_gene_panel(model_var_names, hvg_obs, hvg_parse)
    logger.info(
        "Gene panel: %d (model=%d, hvg_obs=%d, hvg_parse=%d)",
        len(panel),
        len(model_var_names),
        len(hvg_obs),
        len(hvg_parse),
    )
    return panel


def _gene_positions(full_var_names: pd.Index, gene_panel: pd.Index) -> list[int]:
    lookup = {g: i for i, g in enumerate(map(str, full_var_names))}
    return [lookup[g] for g in map(str, gene_panel)]


def _slice_tensor(param: torch.Tensor, positions: list[int], *, axis: int) -> torch.Tensor:
    idx = torch.as_tensor(positions, dtype=torch.long, device=param.device)
    return torch.index_select(param, axis, idx)


def slice_module_to_genes(module: torch.nn.Module, positions: list[int]) -> None:
    n_genes = len(positions)
    with torch.no_grad():
        if hasattr(module, "px_r"):
            px_r = module.px_r
            if px_r.ndim == 1 and px_r.shape[0] > n_genes:
                module.px_r = torch.nn.Parameter(_slice_tensor(px_r, positions, axis=0))
        for name, param in module.named_parameters():
            if param.ndim != 2:
                if "decoder" in name and "bias" in name and param.shape[0] > n_genes:
                    param.copy_(_slice_tensor(param.data, positions, axis=0))
                continue
            if "encoder.fc_layers.Layer 0.0.weight" in name and param.shape[1] > n_genes:
                param.copy_(_slice_tensor(param.data, positions, axis=1))
            elif "decoder" in name and "weight" in name and param.shape[0] > n_genes:
                param.copy_(_slice_tensor(param.data, positions, axis=0))
    if hasattr(module, "n_input") and module.n_input != n_genes:
        module.n_input = n_genes


def slice_model_to_genes(
    model: object,
    gene_panel: pd.Index,
    full_var_names: pd.Index,
) -> pd.Index:
    panel = pd.Index(map(str, gene_panel))
    full = pd.Index(map(str, full_var_names))
    if len(panel) == len(full) and panel.equals(full):
        return panel
    missing = panel[~panel.isin(full)]
    if len(missing):
        raise ValueError(
            f"{len(missing)} panel genes absent from model vars, e.g. {missing[:3].tolist()}"
        )
    positions = _gene_positions(full, panel)
    slice_module_to_genes(model.module, positions)
    m0 = getattr(model, "m0_model", None)
    if m0 is not None:
        slice_module_to_genes(m0.module, positions)
    m0_mod = getattr(getattr(model, "module", None), "m0", None)
    if m0_mod is not None:
        slice_module_to_genes(m0_mod, positions)
    logger.info("Sliced model genes: %d -> %d", len(full), len(panel))
    return panel


if __name__ == "__main__":
    x = sp.random(200, 50, density=0.1, format="csr", random_state=0)
    tiny = ad.AnnData(X=x, var=pd.DataFrame(index=[f"g{i}" for i in range(50)]))
    p = Path("_genes_selfcheck.h5ad")
    tiny.write_h5ad(p)
    try:
        picked = gpu_hvg_genes(p, block_rows=50, n_top_genes=10)
        assert len(picked) > 0, "GPU HVG self-check produced empty gene set"
    finally:
        p.unlink(missing_ok=True)
