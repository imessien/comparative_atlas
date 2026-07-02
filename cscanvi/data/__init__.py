"""Backed atlas I/O for cSCANVI.

Two stacks, one gene panel:

**CuPy (``_genes``)** — HVG selection and model weight slicing to a gene subset.

**PyTorch / Lightning (``_streaming``)** — backed h5ad access, AnnTorchDataset row batches,
BackedScanviDataModule DataLoaders.

Typical perturbation flow::

    panel = resolve_parse_gene_panel(obs_h5ad, parse_h5ad, model_vars, parse_var_slice=...)
    view = setup_backed_anndata(SCANVI, parse_h5ad, panel, registry)
    slice_model_to_genes(model, panel, full_var)
    # TTA + embed: BackedScanviDataModule in cscanvi._ADAPT / _scanvi
"""

from ._genes import (
    gpu_hvg_genes,
    resolve_gene_panel,
    resolve_parse_gene_panel,
    select_hvg,
    slice_model_to_genes,
)
from ._manager import AnnDataManager
from ._streaming import (
    AnnTorchDataset,
    BackedScanviDataModule,
    EmbeddingBatchDataset,
    TTALightningModule,
    close_h5ad,
    collate_scvi_batch,
    contiguous_batch_sampler,
    materialize_backed_slice,
    open_backed,
    open_backed_gene_view,
    read_backed,
    registry_slice_for_inference,
    setup_backed_anndata,
    stratified_pick_indices,
    verify_gene_panel,
)

__all__ = [
    "AnnDataManager",
    "AnnTorchDataset",
    "BackedScanviDataModule",
    "EmbeddingBatchDataset",
    "TTALightningModule",
    "close_h5ad",
    "collate_scvi_batch",
    "contiguous_batch_sampler",
    "gpu_hvg_genes",
    "materialize_backed_slice",
    "open_backed",
    "open_backed_gene_view",
    "read_backed",
    "registry_slice_for_inference",
    "resolve_gene_panel",
    "resolve_parse_gene_panel",
    "select_hvg",
    "setup_backed_anndata",
    "slice_model_to_genes",
    "stratified_pick_indices",
    "verify_gene_panel",
]
