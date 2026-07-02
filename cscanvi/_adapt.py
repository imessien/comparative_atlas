from __future__ import annotations

import logging
from collections import namedtuple
from copy import deepcopy
from pathlib import Path
from typing import Optional, Union

import lightning as L
import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from scvi.data._constants import _SETUP_ARGS_KEY, _SETUP_METHOD_NAME
from scvi.model._utils import parse_device_args
from scvi.model.base._save_load import _initialize_model, _load_saved_files, _validate_var_names
from torch.linalg import vector_norm

from ._scanvae import SCANVAE
from ._scanvi import SCANVI, _device_from_use_gpu
from .data import BackedScanviDataModule, TTALightningModule, setup_backed_anndata

logger = logging.getLogger(__name__)

AdaptHeadLoss = namedtuple(
    "AdaptHeadLoss",
    ["loss", "reconstruction_loss", "energy_score_loss"],
    defaults=(None, None, None),
)


class Adapt(SCANVAE):
    """Energy TTA: align reference SCANVI latents to Parse gene counts via ``m0`` encoder.

    ``X`` is backed Parse counts; ``embedding`` is the reference latent target; loss is
    the energy score between target and ``m0`` encoder ``z``.
    """

    def __init__(self, m0_module: SCANVAE, n_latent: int, **model_kwargs):
        model_kwargs = dict(model_kwargs)
        model_kwargs.setdefault("n_batch", getattr(m0_module, "n_batch", 0))
        model_kwargs.setdefault("n_labels", max(1, getattr(m0_module, "n_labels", 1)))
        model_kwargs.setdefault("n_latent", n_latent)
        model_kwargs.setdefault("dispersion", getattr(m0_module, "dispersion", "gene"))
        model_kwargs.setdefault(
            "gene_likelihood", getattr(m0_module, "gene_likelihood", "zinb")
        )
        super().__init__(
            n_input=n_latent,
            n_hidden=model_kwargs.pop("n_hidden", 128),
            n_layers=model_kwargs.pop("n_layers", 2),
            dropout_rate=model_kwargs.pop("dropout_rate", 0.1),
            **model_kwargs,
        )
        self.init_params_ = self._get_init_params(locals())
        self.was_pretrained = False
        self.m0 = m0_module

    def _get_init_params(self, locals_):
        return {k: v for k, v in locals_.items() if k != "self"}

    def _get_inference_input(self, tensors):
        return self.m0._get_inference_input(tensors)

    def inference(self, *args, **kwargs):
        return self.m0.inference(*args, **kwargs)

    def _get_generative_input(self, tensors, inference_outputs, **kwargs):
        return self.m0._get_generative_input(tensors, inference_outputs, **kwargs)

    def generative(self, *args, **kwargs):
        return self.m0.generative(*args, **kwargs)

    def _adaptation_head_loss(self, tensors, m0_inference_outputs):
        if "embedding" not in tensors:
            raise KeyError(
                "Adaptation loss expected `embedding` in tensors (reference SCANVI latent). "
                f"Available keys: {list(tensors.keys())}"
            )
        z_target = tensors["embedding"]
        m0_z = m0_inference_outputs["z"]
        if z_target.shape[-1] != m0_z.shape[-1]:
            raise ValueError(
                "Latent dim mismatch for energy TTA: "
                f"target embedding {tuple(z_target.shape)} vs m0 z {tuple(m0_z.shape)}."
            )
        if not any(p.requires_grad for p in self.m0.parameters()):
            m0_z = m0_z.detach()
        energy_score_loss = self.energy_loss(z_target, m0_z, verbose=False)
        loss = energy_score_loss.mean()
        zero = torch.zeros((), device=loss.device, dtype=loss.dtype)
        return AdaptHeadLoss(
            loss=loss,
            reconstruction_loss=zero,
            energy_score_loss=energy_score_loss.mean(),
        )

    def _replay_forward(self, *args, **kwargs):
        raise NotImplementedError(
            "Adapt uses TTA_SCANVI.train_test_time_adaptation(); "
            "SCANVI Lightning replay is not supported."
        )

    def vectorize(self, x, multichannel=False):
        if len(x.shape) == 1:
            return x.unsqueeze(1)
        if len(x.shape) == 2:
            return x
        if not multichannel:
            return x.reshape(x.shape[0], -1)
        return x.reshape(x.shape[0], x.shape[1], -1)

    def energy_loss(self, x_true, x_est, beta=1, verbose=True):
        if isinstance(beta, torch.Tensor):
            beta_val = beta.item()
        else:
            beta_val = float(beta)
        eps = 0 if beta_val.is_integer() else 1e-5
        x_true = self.vectorize(x_true).unsqueeze(1)
        if not isinstance(x_est, list):
            n = x_true.shape[0]
            m = x_est.shape[0] // n
            x_est = list(torch.split(x_est, n, dim=0))
        m = len(x_est)
        x_est = [self.vectorize(xi).unsqueeze(1) for xi in x_est]
        x_est = torch.cat(x_est, dim=1)
        s1 = (vector_norm(x_est - x_true, 2, dim=2) + eps).pow(beta).mean(dim=1)
        if m <= 1:
            s2 = torch.zeros_like(s1)
        else:
            dists = torch.cdist(x_est, x_est, p=2) + eps
            s2 = dists.pow(beta).mean(dim=(1, 2)) * m / (m - 1)
        loss = s1 - s2 / 2
        if verbose:
            return loss, s1, s2
        return loss


class TTA_SCANVI(SCANVI):
    """Energy-only test-time adaptation for Parse PBS cells on backed HDF5 gene counts."""

    def __init__(
        self,
        adata: AnnData,
        m0_model: SCANVI,
        adapt_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(adata, **kwargs)
        self.m0_model = m0_model
        adapt_kwargs = adapt_kwargs or {}
        n_latent = int(getattr(m0_model.module, "n_latent", 10))
        self.module = Adapt(m0_module=self.m0_model.module, n_latent=n_latent, **adapt_kwargs)
        self.was_pretrained = False

    @classmethod
    def from_trained_scanvi(cls, reference_model: SCANVI, adapt_kwargs: Optional[dict] = None):
        reference_model._check_if_trained(warn=False)
        model = deepcopy(reference_model)
        model.__class__ = cls
        model.m0_model = reference_model
        adapt_kwargs = adapt_kwargs or {}
        n_latent = int(reference_model.module.n_latent)
        model.module = Adapt(
            m0_module=deepcopy(reference_model.module),
            n_latent=n_latent,
            **adapt_kwargs,
        )
        model.was_pretrained = True
        return model

    def _latent_encoder_module(self):
        return self.module.m0

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
        del kwargs
        if x_adapt_key is not None and x_adapt_key != embedding_key:
            raise ValueError(
                "x_adapt_key differs from embedding_key; Parse energy TTA uses gene counts "
                "from X and reference SCANVI latents only."
            )
        if not isinstance(self.module, Adapt):
            raise TypeError("train_test_time_adaptation expects an Adapt module.")

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
                    f"Missing obsm[{embedding_key!r}] and no embedding array passed."
                )
            emb = np.asarray(adata.obsm[embedding_key])
            embedding = np.asarray(emb[row_index], dtype=np.float32)
        else:
            embedding = np.asarray(embedding, dtype=np.float32)

        n_latent = int(self.module.m0.n_latent)
        if embedding.ndim != 2 or embedding.shape[1] != n_latent:
            raise ValueError(
                f"embedding must be (n_cells, {n_latent}); got {embedding.shape}."
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

        self.module.eval()
        self.is_trained_ = True
        history = {"tta": {"train_loss": [float(trainer.callback_metrics.get("train_loss", 0.0))]}}
        if self.history_ is None:
            self.history_ = history
        else:
            self.history_.update(history)
        return history

    @staticmethod
    def ensure_energy_embedding_backed(
        reference: SCANVI,
        adata: AnnData,
        row_index: np.ndarray,
        *,
        batch_size: int | None = None,
        use_gpu: Union[bool, str, int, None] = None,
    ) -> np.ndarray:
        """Reference SCANVI latents for backed Parse PBS adapt rows."""
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
        """Train Parse energy TTA from backed h5ad (gene counts + reference latents)."""
        view = setup_backed_anndata(SCANVI, h5ad_path, gene_panel, registry)
        n_latent = int(reference.module.n_latent)
        tta = cls.from_trained_scanvi(reference, adapt_kwargs={"n_latent": n_latent})
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
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tta.save(str(out_dir), overwrite=True, save_anndata=False)
        return tta

    @classmethod
    def load_inference(
        cls,
        dir_path: str,
        adata: AnnData,
        *,
        device: int | str = "auto",
    ):
        """Load saved energy-TTA weights for embedding."""
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
            n_latent=n_latent,
        )
        sd = {k: v for k, v in model_state_dict.items() if "ewc_snap_" not in k}
        sd.pop("pyro_param_store", None)
        model.module.load_state_dict(sd, strict=True)
        model.__class__ = cls
        model.is_trained_ = True
        model.module.eval()
        model.to_device(torch_device)
        model._validate_anndata(adata)
        return model
