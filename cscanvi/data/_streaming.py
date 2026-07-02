"""PyTorch / Lightning backed streaming: gene-sliced HDF5 rows → DataLoader.

CuPy HVG and model gene slicing live in ``_genes.py``.
This module is the inference stack:

  setup_backed_anndata  →  AnnTorchDataset (HDF5 row slices)
                       →  BackedScanviDataModule (torch DataLoader)
                       →  TTALightningModule (TTA) or SCANVI.embed_latent_from_backed
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import anndata as ad
import h5py
import lightning as L
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from anndata.abc import CSCDataset, CSRDataset
from scvi import REGISTRY_KEYS
from scvi.data._constants import _SETUP_ARGS_KEY, _SETUP_METHOD_NAME
from torch.utils.data import DataLoader, Dataset, Subset

from ._utils import registry_key_to_default_dtype, scipy_to_torch_sparse

logger = logging.getLogger(__name__)

SparseDataset = (CSRDataset, CSCDataset)


def close_h5ad(adata: ad.AnnData | None) -> None:
    if adata is not None and getattr(adata, "file", None) is not None:
        adata.file.close()


def read_backed(h5ad_path: str | Path) -> ad.AnnData:
    return ad.read_h5ad(h5ad_path, backed="r")


@contextmanager
def open_backed(h5ad_path: str | Path):
    path = Path(h5ad_path)
    if not path.is_file():
        raise FileNotFoundError(f"h5ad not found: {path}")
    backed = read_backed(path)
    try:
        yield backed
    finally:
        close_h5ad(backed)


def verify_gene_panel(query_genes: pd.Index | list[str], var_names: pd.Index | list[str]) -> pd.Index:
    genes = pd.Index(map(str, query_genes))
    missing = genes[~genes.isin(map(str, var_names))]
    if len(missing):
        raise ValueError(
            f"{len(missing)} query genes absent from target (e.g. {missing[:3].tolist()})"
        )
    return genes


def stratified_pick_indices(
    labels: pd.Series,
    n_pick: int,
    min_per_label: int,
    *,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_obs, n_pick = len(labels), min(int(n_pick), len(labels))
    if n_pick >= n_obs:
        return np.arange(n_obs, dtype=np.int64)
    labels_arr = labels.astype(str).to_numpy()
    selected: list[int] = []
    budget = n_pick
    for label in np.unique(labels_arr):
        if budget <= 0:
            break
        inds_arr = np.flatnonzero(labels_arr == label)
        take = min(int(min_per_label), len(inds_arr), budget)
        if take > 0:
            selected.extend(rng.choice(inds_arr, size=take, replace=False).tolist())
            budget -= take
    if budget > 0:
        pool = np.setdiff1d(np.arange(n_obs, dtype=np.int64), np.asarray(selected, dtype=np.int64))
        if len(pool) > 0:
            selected.extend(rng.choice(pool, size=min(budget, len(pool)), replace=False).tolist())
    return np.asarray(selected, dtype=np.int64)


def materialize_backed_slice(
    backed: ad.AnnData,
    row_indices: np.ndarray,
    gene_names: pd.Index | np.ndarray,
    *,
    dense: bool = False,
    batch_key: str | None = None,
    reference_tag: str | None = None,
) -> ad.AnnData:
    genes = pd.Index(map(str, gene_names))
    present = genes[genes.isin(backed.var_names)]
    if len(present) == 0:
        raise ValueError("No gene panel vars present in backed object.")
    out = backed[np.asarray(row_indices, dtype=np.int64), present].to_memory()
    if dense and sp.issparse(out.X):
        out.X = out.X.toarray()
    if batch_key:
        out.obs[batch_key] = "parse"
    if reference_tag:
        out.obs["_comparative_reference"] = reference_tag
    out.obs_names_make_unique()
    return out


def registry_slice_for_inference(
    adata: ad.AnnData,
    gene_names: pd.Index,
    *,
    row_index: np.ndarray | None = None,
    labels_key: str = "cell_type",
    n_pick: int = 4000,
    seed: int = 0,
) -> ad.AnnData:
    present = gene_names[gene_names.isin(adata.var_names)]
    pool = np.asarray(row_index, dtype=np.int64) if row_index is not None else np.arange(adata.n_obs, dtype=np.int64)
    n_pick = min(n_pick, int(pool.size))
    backed = getattr(adata, "isbacked", False) or getattr(adata, "is_view", False)
    if n_pick >= pool.size and not backed:
        out = adata[pool, present].copy() if row_index is not None else adata[:, present].copy()
    else:
        if labels_key in adata.obs.columns:
            labels = adata.obs[labels_key].iloc[pool]
            local = stratified_pick_indices(labels, n_pick, max(1, 3), seed=seed)
        else:
            local = np.random.default_rng(seed).choice(pool.size, min(n_pick, pool.size), replace=False)
        pick = pool[local]
        sub = adata[pick, present]
        out = sub.to_memory() if backed else sub.copy()
    if labels_key in adata.obs.columns:
        src = adata.obs[labels_key].iloc[pool] if row_index is not None else adata.obs[labels_key]
        out.uns["_knn_full_label_counts"] = src.astype(str).value_counts()
    return out


def open_backed_gene_view(h5ad_path: str | Path, gene_panel: pd.Index) -> ad.AnnData:
    backed = ad.read_h5ad(h5ad_path, backed="r")
    present = pd.Index([g for g in map(str, gene_panel) if g in backed.var_names])
    if len(present) == 0:
        if getattr(backed, "file", None) is not None:
            backed.file.close()
        raise ValueError(f"No gene panel vars present in {h5ad_path}")
    return backed[:, present]


def setup_backed_anndata(
    model_cls: type,
    h5ad_path: str | Path,
    gene_panel: pd.Index,
    registry: dict,
) -> ad.AnnData:
    view = open_backed_gene_view(h5ad_path, gene_panel)
    setup_args = dict(registry[_SETUP_ARGS_KEY])
    method_name = registry.get(_SETUP_METHOD_NAME, "setup_anndata")
    getattr(model_cls, method_name)(
        view,
        source_registry=registry,
        extend_categories=True,
        allow_missing_labels=True,
        **setup_args,
    )
    return view


class AnnTorchDataset(Dataset):
    """torch Dataset that slices rows from a backed AnnData (HDF5) on __getitem__."""

    def __init__(
        self,
        adata_manager: AnnDataManager,
        getitem_tensors: list | dict[str, type] | None = None,
        load_sparse_tensor: bool = False,
    ):
        super().__init__()
        if adata_manager.adata is None:
            raise ValueError("Please run ``register_fields`` on ``adata_manager`` first.")
        self.adata_manager = adata_manager
        self.keys_and_dtypes = getitem_tensors
        self.load_sparse_tensor = load_sparse_tensor

    @property
    def registered_keys(self):
        return self.adata_manager.data_registry.keys()

    @property
    def keys_and_dtypes(self):
        return self._keys_and_dtypes

    @keys_and_dtypes.setter
    def keys_and_dtypes(self, getitem_tensors: list | dict[str, type] | None):
        if isinstance(getitem_tensors, list):
            keys_to_dtypes = {key: registry_key_to_default_dtype(key) for key in getitem_tensors}
        elif isinstance(getitem_tensors, dict):
            keys_to_dtypes = getitem_tensors
        elif getitem_tensors is None:
            keys_to_dtypes = {
                key: registry_key_to_default_dtype(key) for key in self.registered_keys
            }
        else:
            raise ValueError("`getitem_tensors` must be a `list`, `dict`, or `None`")
        for key in keys_to_dtypes:
            if key not in self.registered_keys:
                raise KeyError(f"{key} not found in the data registry.")
        self._keys_and_dtypes = keys_to_dtypes

    @property
    def data(self):
        if not hasattr(self, "_data"):
            self._data = {
                key: self.adata_manager.get_from_registry(key) for key in self.keys_and_dtypes
            }
        return self._data

    def __len__(self):
        return self.adata_manager.adata.shape[0]

    def __getitem__(self, indexes: int | list[int] | slice) -> dict[str, np.ndarray | torch.Tensor]:
        single_idx = isinstance(indexes, (int, np.integer))
        if single_idx:
            indexes = [int(indexes)]
        if self.adata_manager.adata.isbacked and isinstance(indexes, list | np.ndarray):
            indexes = np.sort(indexes)
        data_map = {}
        for key, dtype in self.keys_and_dtypes.items():
            data = self.data[key]
            if isinstance(data, np.ndarray | h5py.Dataset):
                sliced_data = data[indexes].astype(dtype, copy=False)
            elif isinstance(data, pd.DataFrame):
                sliced_data = data.iloc[indexes, :].to_numpy().astype(dtype, copy=False)
            elif sp.issparse(data) or isinstance(data, SparseDataset):
                sliced_data = data[indexes].astype(dtype, copy=False)
                if self.load_sparse_tensor:
                    sliced_data = scipy_to_torch_sparse(sliced_data)
                else:
                    sliced_data = sliced_data.toarray()
            elif isinstance(data, str) and key == REGISTRY_KEYS.MINIFY_TYPE_KEY:
                continue
            else:
                raise TypeError(f"{key} is not a supported type")
            if single_idx and isinstance(sliced_data, np.ndarray) and sliced_data.ndim > 1 and sliced_data.shape[0] == 1:
                sliced_data = np.squeeze(sliced_data, axis=0)
            data_map[key] = sliced_data
        return data_map


def collate_scvi_batch(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key in batch[0]:
        vals = [b[key] for b in batch]
        first = vals[0]
        if isinstance(first, np.ndarray):
            arr = np.stack(vals, axis=0)
            dtype = torch.float32 if arr.dtype.kind == "f" else torch.int64
            out[key] = torch.as_tensor(arr, dtype=dtype)
        elif isinstance(first, torch.Tensor):
            out[key] = torch.stack(vals, dim=0)
        else:
            out[key] = torch.as_tensor(np.asarray(vals))
        if key in (REGISTRY_KEYS.X_KEY, "X") and out[key].ndim == 3 and out[key].shape[1] == 1:
            out[key] = out[key].squeeze(1)
    return out


class EmbeddingBatchDataset(Dataset):
    def __init__(
        self,
        base: AnnTorchDataset | Subset,
        embedding: np.ndarray | None,
        subset_positions: np.ndarray | None = None,
    ):
        self.base = base
        self.embedding = None if embedding is None else np.asarray(embedding, dtype=np.float32)
        self.subset_positions = subset_positions

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.base[idx]
        if self.embedding is not None:
            pos = int(self.subset_positions[idx]) if self.subset_positions is not None else idx
            item = dict(item)
            item["embedding"] = self.embedding[pos]
        return item


def contiguous_batch_sampler(row_index: np.ndarray, batch_size: int) -> Iterator[list[int]]:
    if row_index.size == 0:
        return
    order = np.argsort(row_index)
    sorted_rows = row_index[order]
    run_start = 0
    for i in range(1, len(sorted_rows) + 1):
        if i == len(sorted_rows) or sorted_rows[i] != sorted_rows[i - 1] + 1:
            run_positions = order[run_start:i].tolist()
            for j in range(0, len(run_positions), batch_size):
                yield run_positions[j : j + batch_size]
            run_start = i


class BackedScanviDataModule(L.LightningDataModule):
    def __init__(
        self,
        adata_manager: AnnDataManager,
        row_index: np.ndarray,
        *,
        embedding: np.ndarray | None = None,
        batch_size: int = 128,
        num_workers: int = 0,
        shuffle: bool = False,
        tensor_keys: Sequence[str] | None = None,
    ):
        super().__init__()
        self.adata_manager = adata_manager
        self.row_index = np.asarray(row_index, dtype=np.int64)
        self.embedding = embedding
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.tensor_keys = list(tensor_keys) if tensor_keys is not None else [
            REGISTRY_KEYS.X_KEY,
            REGISTRY_KEYS.BATCH_KEY,
            REGISTRY_KEYS.LABELS_KEY,
        ]
        self._dataset: EmbeddingBatchDataset | None = None
        self._subset_positions = np.arange(len(self.row_index), dtype=np.int64)

    def setup(self, stage: str | None = None) -> None:
        base = self.adata_manager.create_torch_dataset(
            indices=self.row_index.tolist(),
            data_and_attributes=self.tensor_keys,
        )
        self._dataset = EmbeddingBatchDataset(
            base,
            self.embedding,
            subset_positions=self._subset_positions,
        )

    def _loader(self, shuffle: bool) -> DataLoader:
        if self._dataset is None:
            self.setup()
        return DataLoader(
            self._dataset,
            batch_size=self.batch_size,
            shuffle=shuffle and self.shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_scvi_batch,
            drop_last=True,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(shuffle=True)

    def predict_dataloader(self) -> DataLoader:
        return self._loader(shuffle=False)


class TTALightningModule(L.LightningModule):
    def __init__(
        self,
        tta_model: Any,
        *,
        lr: float = 1e-3,
        m0_lr: float = 1e-4,
        weight_decay: float = 1e-6,
        energy_only: bool = True,
    ):
        super().__init__()
        self.tta_model = tta_model
        self.adapt_module = tta_model.module
        self.lr = lr
        self.m0_lr = m0_lr
        self.weight_decay = weight_decay
        self.energy_only = energy_only
        self._configure_trainable()

    def _configure_trainable(self) -> None:
        mod = self.adapt_module
        for p in mod.m0.parameters():
            p.requires_grad_(True)
        for name, p in mod.named_parameters():
            if name.startswith("z_encoder.") or name.startswith("l_encoder."):
                p.requires_grad_(False)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        mod = self.adapt_module
        tensors = {k: v.to(self.device) for k, v in batch.items()}
        m0_in = mod.m0._get_inference_input(tensors)
        with torch.set_grad_enabled(any(p.requires_grad for p in mod.m0.parameters())):
            m0_out = mod.m0.inference(**m0_in)
        losses = mod._adaptation_head_loss(tensors, m0_out)
        loss = losses.loss if losses.loss.ndim == 0 else losses.loss.mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite TTA loss at batch {batch_idx}")
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("energy_score_loss", losses.energy_score_loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        mod = self.adapt_module
        adapt_params = [
            p for name, p in mod.named_parameters()
            if p.requires_grad and not name.startswith("m0.")
        ]
        m0_params = [p for p in mod.m0.parameters() if p.requires_grad]
        groups = []
        if adapt_params:
            groups.append({"params": adapt_params, "lr": self.lr, "weight_decay": self.weight_decay})
        if m0_params:
            groups.append({"params": m0_params, "lr": self.m0_lr, "weight_decay": self.weight_decay})
        return torch.optim.Adam(groups, eps=0.01)
