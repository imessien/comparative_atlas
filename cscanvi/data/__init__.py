"""Backed atlas I/O for cSCANVI.

Typical Parse flow::

    panel = resolve_parse_gene_panel(obs_h5ad, parse_h5ad, model_vars)
    view = setup_backed_anndata(SCANVI, parse_h5ad, panel, registry)
    slice_model_to_genes(model, panel, full_var)
    # TTA + embed via BackedScanviDataModule in cscanvi._adapt
"""

from ._genes import resolve_parse_gene_panel, slice_model_to_genes
from ._streaming import (
    BackedScanviDataModule,
    TTALightningModule,
    close_h5ad,
    materialize_backed_slice,
    open_backed,
    open_backed_gene_view,
    read_backed,
    setup_backed_anndata,
    stratified_pick_indices,
    verify_gene_panel,
)

__all__ = [
    "BackedScanviDataModule",
    "TTALightningModule",
    "close_h5ad",
    "materialize_backed_slice",
    "open_backed",
    "open_backed_gene_view",
    "read_backed",
    "resolve_parse_gene_panel",
    "setup_backed_anndata",
    "slice_model_to_genes",
    "stratified_pick_indices",
    "verify_gene_panel",
]
