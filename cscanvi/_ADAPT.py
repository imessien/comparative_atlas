import logging
from copy import deepcopy
from pathlib import Path
from typing import Optional, Sequence, Union

import lightning as L
import numpy as np
import pandas as pd
import torch
from anndata import AnnData

from scvi.data._constants import _SETUP_ARGS_KEY, _SETUP_METHOD_NAME
from scvi.model._utils import parse_device_args
from scvi.model.base._save_load import _initialize_model, _load_saved_files, _validate_var_names

from ._adapt import Adapt
from ._scanvi import SCANVI, _device_from_use_gpu
from .data import BackedScanviDataModule, TTALightningModule, setup_backed_anndata, slice_model_to_genes



# from scvi.model._scvi import SCVI
# from scvi.model.base import ArchesMixin, BaseModelClass, RNASeqMixin, VAEMixin

logger = logging.getLogger(__name__)


class TTA_SCANVI(SCANVI):
    """Energy-only test-time adaptation for Parse and observational SCANVI.

  Parse 10M workflow (``perturbation.build_parse_tta_model``):

    1. ``ensure_energy_embedding_backed`` — reference SCANVI latents for PBS adapt rows.
    2. ``run_energy_tta_backed`` — stream Parse gene counts + latents; align ``m0`` encoder.
    3. ``load_inference`` + ``embed_latent_from_backed`` — embed matched Parse cells from counts.

    External embeddings (scGPT, etc.) require ``energy_only=False`` and a projection head
    when ``n_input != n_latent``.
    """

    def __init__(
        self,
        adata: AnnData,
        m0_model: SCANVI,
        adapt_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """Initialize from explicit adata + scanvi model.

        Note
        ----
        For test-time adaptation of an already trained continual model,
        prefer :meth:`from_trained_scanvi`, which reuses the trained model's
        registry/keys directly.
        """
        super().__init__(adata, **kwargs)
        self.m0_model = m0_model

        adapt_kwargs = adapt_kwargs or {}
        self.module = Adapt(m0_module=self.m0_model.module, **adapt_kwargs)

        self.was_pretrained = False

    @classmethod
    def from_trained_scanvi(
        cls,
        reference_model: SCANVI,
        adapt_kwargs: Optional[dict] = None,
    ):
        """Create TTA model directly from a trained SCANVI model.

        This avoids passing/setup of ``labels_key``, ``unlabeled_category``,
        ``batch_key`` and other setup arguments: all tensor registration metadata
        is inherited from the trained reference model.
        """
        reference_model._check_if_trained(warn=False)
        model = deepcopy(reference_model)
        model.__class__ = cls
        model.m0_model = reference_model
        adapt_kwargs = adapt_kwargs or {}
        model.module = Adapt(m0_module=deepcopy(reference_model.module), **adapt_kwargs)
        model.was_pretrained = True
        return model

    def _latent_encoder_module(self):
        mod = self.module
        if isinstance(mod, Adapt) and not mod.use_embedding_for_inference:
            return mod.m0
        return mod

    @staticmethod
    def _init_adapt_epoch_running():
        return {
            "train_loss": 0.0,
            "adapt_reconstruction_loss": 0.0,
            "energy_score_loss": 0.0,
        }

    @staticmethod
    def _update_adapt_epoch_running(running, losses):
        total = losses.loss if losses.loss.ndim == 0 else losses.loss.mean()
        running["train_loss"] += float(total.detach().cpu())
        running["adapt_reconstruction_loss"] += float(
            losses.reconstruction_loss.detach().cpu()
        )
        running["energy_score_loss"] += float(losses.energy_score_loss.detach().cpu())

    @staticmethod
    def _finalize_adapt_epoch_running(running, n_batches):
        denom = max(n_batches, 1)
        return {key: value / denom for key, value in running.items()}

    def train_test_time_adaptation(
        self,
        adata: AnnData,
        embedding_key: str,
        *,
        max_epochs: Optional[int] = None,
        batch_size: int = 128,
        train_size: float = 1.0,
        use_gpu: Optional[Union[str, int, bool]] = None,
        plan_kwargs: Optional[dict] = None,
        row_index: np.ndarray | None = None,
        embedding: np.ndarray | None = None,
        x_adapt_key: str | None = None,
        **kwargs,
    ):
        """Energy TTA via backed streaming DataLoader (Parse gene counts + reference latents).

        ``embedding_key`` / ``x_adapt_key`` name the target latent for in-memory ``obsm``
        when ``embedding`` is not passed. Parse PBS adapt passes ``embedding`` directly
        from ``ensure_energy_embedding_backed``; gene counts always come from the registered
        ``X`` tensor (backed h5ad rows), not from ``obsm``.
        """
        del kwargs  # ponytail: legacy scvi kwargs unused in Lightning TTA path
        if x_adapt_key is not None and x_adapt_key != embedding_key:
            raise ValueError(
                "x_adapt_key differs from embedding_key; Parse energy TTA uses gene counts "
                "from X and reference SCANVI latents only."
            )
        if not isinstance(self.module, Adapt):
            raise TypeError(
                "train_test_time_adaptation expects a TTA_SCANVI model with an "
                "Adapt module (energy TTA)."
            )

        adata = self._validate_anndata(adata)
        adata_manager = self.get_anndata_manager(adata, required=True)
        device = _device_from_use_gpu(use_gpu)
        logger.info("TTA device: %s", device)

        if row_index is None:
            row_index = np.arange(adata.n_obs, dtype=np.int64)
        else:
            row_index = np.asarray(row_index, dtype=np.int64)

        if embedding is None:
            if embedding_key not in adata.obsm:
                raise KeyError(
                    f"Missing obsm[{embedding_key!r}] and no embedding array passed. "
                    "Parse adapt should pass embedding= from ensure_energy_embedding_backed."
                )
            emb = np.asarray(adata.obsm[embedding_key])
            embedding = np.asarray(emb[row_index], dtype=np.float32)
        else:
            embedding = np.asarray(embedding, dtype=np.float32)

        n_latent = int(getattr(self.module.m0, "n_latent", self.module.n_latent))
        if embedding.ndim != 2 or embedding.shape[1] != n_latent:
            raise ValueError(
                f"embedding must be (n_cells, {n_latent}); got {embedding.shape}. "
                "Parse PBS adapt expects reference SCANVI latents, not external embeddings."
            )

        if max_epochs is None:
            max_epochs = 20
        plan_kwargs = dict(plan_kwargs or {})
        lr = float(plan_kwargs.get("lr", 1e-3))
        m0_lr = float(plan_kwargs.get("m0_lr", lr))
        weight_decay = float(plan_kwargs.get("weight_decay", 1e-6))

        train_n = int(np.floor(train_size * row_index.size))
        if train_n >= row_index.size:
            sel = np.arange(row_index.size)
        else:
            sel = np.random.permutation(row_index.size)[:train_n]
        train_rows = row_index[sel]
        train_emb = embedding[sel]
        if train_rows.size == 0:
            raise ValueError("Test-time adaptation train split is empty; increase `train_size`.")

        self.to_device(device)
        self.module.train()
        # Parse TTA: m0 encodes backed gene counts; embedding is the alignment target only.
        self.module.use_embedding_for_inference = False

        dm = BackedScanviDataModule(
            adata_manager,
            train_rows,
            embedding=train_emb,
            batch_size=batch_size,
            shuffle=True,
        )
        lit = TTALightningModule(
            self,
            lr=lr,
            m0_lr=m0_lr,
            weight_decay=weight_decay,
            energy_only=getattr(self.module, "energy_only", False),
        )
        accelerator = "gpu" if device.type == "cuda" else "cpu"
        trainer = L.Trainer(
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=1,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )
        trainer.fit(lit, datamodule=dm)

        self.module.use_embedding_for_inference = False
        self.module.eval()
        self.is_trained_ = True
        history = {"tta": {"train_loss": [float(trainer.callback_metrics.get("train_loss", 0.0))]}}
        if self.history_ is None:
            self.history_ = history
        else:
            self.history_.update(history)
        return history

    def register_ewc_anchor(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        batch_size: int = 256,
        uniform: bool = False,
    ):
        """Set the EWC anchor for the adaptation module.

        Snapshots the module's current trainable parameters as the EWC anchor
        and computes (Fisher) importances so that subsequent training with
        ``plan_kwargs={"ewc_importance": > 0}`` penalizes drift away from this
        state. This regularizes the *adaptation* module against itself; it does
        not reference or modify the reference ``m0`` weights.

        Call before EWC-regularized training if using legacy panel-decoder mode.

        Parameters
        ----------
        adata
            AnnData to estimate importances on. Defaults to the model's adata.
        indices
            Optional subset of observations to use.
        batch_size
            Minibatch size for the importance estimation loader.
        uniform
            If ``True``, skip Fisher estimation and use uniform (ones)
            importances, i.e. a plain quadratic anchor.
        """
        if uniform:
            self.module.register_ewc_anchor()
            return

        adata = self._validate_anndata(adata)
        dataloader = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )
        importances = self._compute_importances(model=self, dataloader=dataloader)
        self.module.register_ewc_anchor(importances=importances)

    @staticmethod
    def ensure_energy_embedding(
        adata: AnnData,
        reference: SCANVI,
        embedding_key: str,
        *,
        batch_size: int | None = None,
        use_gpu: Union[bool, str, int, None] = None,
        row_index: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Set or return ``obsm[embedding_key]`` from the reference encoder when absent."""
        if row_index is not None:
            return reference.embed_latent_from_backed(
                adata,
                row_index,
                batch_size=batch_size,
                use_gpu=use_gpu,
            )
        if embedding_key in adata.obsm:
            return None
        lat = reference.embed_latent_from_adata(
            adata,
            batch_size=batch_size,
            use_gpu=use_gpu,
        )
        adata.obsm[embedding_key] = np.asarray(lat, dtype=np.float32)
        return lat

    @staticmethod
    def ensure_energy_embedding_backed(
        reference: SCANVI,
        adata: AnnData,
        row_index: np.ndarray,
        *,
        batch_size: int | None = None,
        use_gpu: Union[bool, str, int, None] = None,
    ) -> np.ndarray:
        """Reference SCANVI latents for backed Parse PBS adapt rows (gene-count encoder)."""
        reference._register_manager_for_instance(
            reference.adata_manager.transfer_fields(
                adata,
                extend_categories=True,
                allow_missing_labels=True,
            )
        )
        return reference.embed_latent_from_backed(
            adata,
            row_index,
            batch_size=batch_size,
            use_gpu=use_gpu,
        )

    @classmethod
    def run_energy_tta_backed(
        cls,
        reference: SCANVI,
        *,
        h5ad_path: str | Path,
        row_index: np.ndarray,
        gene_panel,
        embedding: np.ndarray,
        registry: dict,
        out_dir: str | Path,
        embedding_key: str = "X_tta_energy",
        max_epochs: int = 20,
        batch_size: int = 128,
        use_gpu: Optional[Union[str, int, bool]] = None,
        plan_kwargs: Optional[dict] = None,
    ) -> "TTA_SCANVI":
        """Train Parse energy TTA from backed h5ad (gene counts + reference SCANVI latents)."""
        view = setup_backed_anndata(SCANVI, h5ad_path, gene_panel, registry)
        full_var = pd.Index(map(str, reference.adata.var_names)) if reference.adata is not None else pd.Index(map(str, gene_panel))
        slice_model_to_genes(reference, gene_panel, full_var)

        n_latent = int(reference.module.n_latent)
        tta = cls.from_trained_scanvi(
            reference,
            adapt_kwargs={
                "n_input": n_latent,
                "n_latent": n_latent,
                "energy_only": True,
            },
        )
        tta._register_manager_for_instance(
            SCANVI._get_most_recent_anndata_manager(view, required=True)
        )
        tta.train_test_time_adaptation(
            view,
            embedding_key=embedding_key,
            max_epochs=max_epochs,
            batch_size=batch_size,
            train_size=1.0,
            use_gpu=use_gpu,
            plan_kwargs=plan_kwargs or {},
            row_index=row_index,
            embedding=embedding,
        )
        tta.module.use_embedding_for_inference = False
        tta.module.eval()
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tta.save(str(out_dir), overwrite=True, save_anndata=False)
        return tta

    @classmethod
    def run_energy_tta(
        cls,
        reference: SCANVI,
        adapt_adata: AnnData,
        *,
        embedding_key: str,
        out_dir: str | Path,
        max_epochs: int = 20,
        batch_size: int = 128,
        use_gpu: Optional[Union[str, int, bool]] = None,
        plan_kwargs: Optional[dict] = None,
    ) -> "TTA_SCANVI":
        """Train energy TTA on ``adapt_adata`` and save to ``out_dir``."""
        n_latent = int(reference.module.n_latent)
        tta = cls.from_trained_scanvi(
            reference,
            adapt_kwargs={
                "n_input": n_latent,
                "n_latent": n_latent,
                "energy_only": True,
            },
        )
        tta._register_manager_for_instance(
            SCANVI._get_most_recent_anndata_manager(adapt_adata, required=True)
        )
        tta.train_test_time_adaptation(
            adapt_adata,
            embedding_key=embedding_key,
            max_epochs=max_epochs,
            batch_size=batch_size,
            train_size=1.0,
            use_gpu=use_gpu,
            plan_kwargs=plan_kwargs or {},
            embedding=np.asarray(adapt_adata.obsm[embedding_key], dtype=np.float32),
        )
        tta.module.use_embedding_for_inference = False
        tta.module.eval()
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tta.save(str(out_dir), overwrite=True, save_anndata=True)
        return tta

    @classmethod
    def load_inference(
        cls,
        dir_path: str,
        adata: AnnData,
        *,
        device: int | str = "auto",
    ):
        """Load saved energy-TTA weights for embedding (no observational model, no EWC)."""
        if isinstance(device, int):
            torch_device = _device_from_use_gpu(device)
        else:
            _, _, torch_device = parse_device_args(
                accelerator="auto",
                devices=device,
                return_device="torch",
                validate_single_device=True,
            )
        attr_dict, var_names, model_state_dict, _ = _load_saved_files(
            dir_path, load_adata=False, map_location=torch_device
        )
        registry = attr_dict.pop("registry_")
        n_latent = int(attr_dict["init_params_"]["non_kwargs"]["n_latent"])
        _validate_var_names(adata, var_names)
        method_name = registry.get(_SETUP_METHOD_NAME, "setup_anndata")
        getattr(SCANVI, method_name)(adata, source_registry=registry, **registry[_SETUP_ARGS_KEY])
        model = _initialize_model(SCANVI, adata, registry, attr_dict, None)
        model.module = Adapt(
            m0_module=deepcopy(model.module),
            n_input=n_latent,
            n_latent=n_latent,
            energy_only=True,
        )
        sd = {k: v for k, v in model_state_dict.items() if "ewc_snap_" not in k}
        sd.pop("pyro_param_store", None)
        model.module.load_state_dict(sd, strict=True)
        model.__class__ = cls
        model.module.use_embedding_for_inference = False
        model.is_trained_ = True
        model.module.eval()
        model.to_device(torch_device)
        model._validate_anndata(adata)
        return model
