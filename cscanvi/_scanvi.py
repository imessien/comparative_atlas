from __future__ import annotations

import logging
import os
import warnings
from copy import deepcopy
from typing import TYPE_CHECKING, Sequence, Union

import numpy as np
import torch
from anndata import AnnData
from scvi import REGISTRY_KEYS
from scvi.data import _constants
from scvi.data._constants import _MODEL_NAME_KEY, _SETUP_ARGS_KEY, _SETUP_METHOD_NAME
from scvi.model._scanvi import SCANVI as _SCANVI
from scvi.model._utils import parse_device_args
from scvi.model.base._archesmixin import _get_loaded_data, _set_params_online_update
from scvi.model.base._save_load import (
    _initialize_model,
    _load_saved_files,
    _validate_var_names,
)
from scvi.model.base._base_model import BaseModelClass

from ._scanvae import SCANVAE
from ._trainingplans import CLSemiSupervisedTrainingPlan
from ._utils import compute_uncertainty_scores
from .data import BackedScanviDataModule

if TYPE_CHECKING:
    from scvi.model.base._base_model import BaseModelClass as _BaseModelClass

logger = logging.getLogger(__name__)


def _embed_batch_size(batch_size: int | None) -> int:
    if batch_size is not None:
        return int(batch_size)
    return int(os.environ.get("PARSE_EMBED_BATCH_SIZE", "1024"))


def _device_from_use_gpu(use_gpu: Union[bool, str, int, None]) -> torch.device:
    if use_gpu is False or use_gpu == "cpu":
        return torch.device("cpu")
    if isinstance(use_gpu, int):
        return torch.device(f"cuda:{use_gpu}") if torch.cuda.is_available() else torch.device("cpu")
    if use_gpu is True and torch.cuda.is_available():
        raw = os.environ.get("CUDA_DEVICE", os.environ.get("LOCAL_RANK", "")).strip()
        if raw:
            return torch.device(f"cuda:{int(raw)}")
    _, _, device = parse_device_args(
        accelerator="auto",
        devices="auto",
        return_device="torch",
        validate_single_device=True,
    )
    return device


class SCANVI(_SCANVI):
    _module_cls = SCANVAE
    _training_plan_cls = CLSemiSupervisedTrainingPlan

    def _latent_encoder_module(self):
        return self.module

    @torch.inference_mode()
    def embed_latent_from_adata(
        self,
        adata: AnnData,
        *,
        batch_size: int | None = None,
        use_gpu: Union[bool, str, int, None] = None,
    ) -> np.ndarray:
        """Encode gene counts to latent via GPU dataloader batches."""
        batch_size = _embed_batch_size(batch_size)
        adata = self._validate_anndata(adata)
        device = _device_from_use_gpu(True if use_gpu is None else use_gpu)
        self.to_device(device)
        self.module.eval()
        encoder = self._latent_encoder_module()
        loader_kwargs: dict = {}
        if device.type == "cuda":
            loader_kwargs["pin_memory"] = True
        parts: list[torch.Tensor] = []
        for tensors in self._make_data_loader(
            adata=adata, batch_size=batch_size, **loader_kwargs
        ):
            tensors = {key: val.to(device, non_blocking=True) for key, val in tensors.items()}
            inf_in = encoder._get_inference_input(tensors)
            z = encoder.inference(**inf_in)["z"]
            parts.append(z.detach())
        if not parts:
            raise ValueError("embed_latent_from_adata: empty dataloader")
        return torch.cat(parts, dim=0).cpu().numpy().astype(np.float32, copy=False)

    def _make_backed_data_loader(
        self,
        adata: AnnData,
        row_index: np.ndarray,
        *,
        batch_size: int,
        shuffle: bool = False,
        embedding: np.ndarray | None = None,
    ):
        mgr = self.get_anndata_manager(adata, required=True)
        dm = BackedScanviDataModule(
            mgr,
            np.asarray(row_index, dtype=np.int64),
            embedding=embedding,
            batch_size=batch_size,
            shuffle=shuffle,
        )
        dm.setup()
        return dm.train_dataloader() if shuffle else dm.predict_dataloader()

    @torch.inference_mode()
    def embed_latent_from_backed(
        self,
        adata: AnnData,
        row_index: np.ndarray,
        *,
        batch_size: int | None = None,
        use_gpu: Union[bool, str, int, None] = None,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        """Encode backed AnnData rows to latent without materializing X."""
        batch_size = _embed_batch_size(batch_size)
        adata = self._validate_anndata(adata)
        row_index = np.asarray(row_index, dtype=np.int64)
        device = _device_from_use_gpu(True if use_gpu is None else use_gpu)
        self.to_device(device)
        self.module.eval()
        encoder = self._latent_encoder_module()
        n_latent = int(encoder.n_latent) if hasattr(encoder, "n_latent") else int(self.module.n_latent)
        latent = out if out is not None else np.empty((row_index.size, n_latent), dtype=np.float32)
        loader = self._make_backed_data_loader(
            adata, row_index, batch_size=batch_size, shuffle=False
        )
        offset = 0
        for tensors in loader:
            tensors = {key: val.to(device, non_blocking=True) for key, val in tensors.items()}
            inf_in = encoder._get_inference_input(tensors)
            z = encoder.inference(**inf_in)["z"]
            n = int(z.shape[0])
            latent[offset : offset + n] = z.detach().cpu().numpy().astype(np.float32, copy=False)
            offset += n
        if offset != row_index.size:
            raise ValueError(f"embed_latent_from_backed: expected {row_index.size} rows, got {offset}")
        return latent

    @torch.inference_mode()
    def embed_query_latent(
        self,
        adata: AnnData,
        *,
        batch_size: int | None = None,
        use_gpu: Union[bool, str, int, None] = None,
        row_index: np.ndarray | None = None,
    ) -> np.ndarray:
        """Register query cells (extend batch/label categories) and encode to latent."""
        self._register_manager_for_instance(
            self.adata_manager.transfer_fields(
                adata,
                extend_categories=True,
                allow_missing_labels=True,
            )
        )
        if row_index is not None:
            return self.embed_latent_from_backed(
                adata,
                row_index,
                batch_size=batch_size,
                use_gpu=use_gpu,
            )
        return self.embed_latent_from_adata(
            adata,
            batch_size=batch_size,
            use_gpu=use_gpu,
        )

    @classmethod
    def load(
        cls,
        dir_path: str,
        adata: AnnData | None = None,
        prefix: str | None = None,
        backup_url: str | None = None,
        accelerator: str = "auto",
        device: int | str = "auto",
        *,
        restore_ewc: bool = True,
        **kwargs,
    ):
        load_adata = adata is None
        _, _, torch_device = parse_device_args(
            accelerator=accelerator,
            devices=device,
            return_device="torch",
            validate_single_device=True,
        )

        (
            attr_dict,
            var_names,
            model_state_dict,
            new_adata,
        ) = _load_saved_files(
            dir_path,
            load_adata,
            map_location=torch_device,
            prefix=prefix,
            backup_url=backup_url,
        )
        adata = new_adata if new_adata is not None else adata

        registry = attr_dict.pop("registry_")
        if _MODEL_NAME_KEY in registry and registry[_MODEL_NAME_KEY] != cls.__name__:
            raise ValueError("It appears you are loading a model from a different class.")

        if adata:
            if _SETUP_ARGS_KEY not in registry:
                raise ValueError(
                    "Saved model does not contain original setup inputs. "
                    "Cannot load the original setup."
                )
            _validate_var_names(adata, var_names)
            method_name = registry.get(_SETUP_METHOD_NAME, "setup_anndata")
            getattr(cls, method_name)(
                adata, source_registry=registry, **registry[_SETUP_ARGS_KEY]
            )

        model = _initialize_model(cls, adata, registry, attr_dict, kwargs.get("datamodule"))
        pyro_param_store = model_state_dict.pop("pyro_param_store", None)
        model.module.on_load(model, pyro_param_store=pyro_param_store)

        if not restore_ewc:
            model_state_dict = {
                k: v for k, v in model_state_dict.items() if "ewc_snap_" not in k
            }
        elif hasattr(model.module, "register_ewc_buffers_from_state_dict"):
            restored = model.module.register_ewc_buffers_from_state_dict(model_state_dict)
            if restored:
                n_ewc = int(model.module.get_buffer("ewc_snap_count").item())
                logger.info("Restoring %d EWC replay snapshot(s) from checkpoint.", n_ewc)

        model.module.load_state_dict(model_state_dict, strict=restore_ewc)
        if restore_ewc and hasattr(model.module, "_ensure_ewc_lists_from_buffers"):
            model.module._ensure_ewc_lists_from_buffers()

        model.to_device(torch_device)
        model.module.eval()
        if adata:
            model._validate_anndata(adata)
        return model

    @classmethod
    def get_uncertainty(
        cls,
        adata: AnnData,
        reference_model: Union[str, _BaseModelClass],
        num_points: int = 10,
        order: str = "top-k",
        indices: Sequence[int] | None = None,
        batch_size: int | None = None,
        tta_rep: int = 10,
    ):
        reference_model._check_if_trained(warn=False)
        adata = reference_model._validate_anndata(adata)
        scdl = reference_model._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )
        from tqdm import tqdm

        unc_scores = []
        device = reference_model.device
        for tensors in tqdm(scdl, desc="cscanvi.get_uncertainty", unit="batch"):
            inference_inputs = reference_model.module._get_inference_input(tensors)
            unc_batch = compute_uncertainty_scores(
                inference_inputs, reference_model, device, tta_rep=tta_rep
            )
            unc_scores.extend(unc_batch)
        if order == "top-k":
            score_idx = torch.sort(torch.tensor(unc_scores), descending=True)[1][:num_points]
        elif order == "bottom-k":
            score_idx = torch.sort(torch.tensor(unc_scores), descending=False)[1][:num_points]
        elif order == "step":
            skip = len(unc_scores) // num_points
            steps = np.arange(0, len(unc_scores), skip)
            score_idx = torch.sort(torch.tensor(unc_scores), descending=True)[1][steps]
        else:
            raise ValueError(
                f"Invalid value for 'order': {order}. Expected 'top-k', 'bottom-k' or 'step'."
            )
        return unc_scores, score_idx

    @classmethod
    def load_query_data_with_replay(
        cls,
        adata: AnnData,
        reference_model: Union[str, BaseModelClass],
        control_uns_key: str | None = None,
        replay_uns_key: str | None = None,
        inplace_subset_query_vars: bool = False,
        use_gpu: Union[bool, str, int, None] = None,
        unfrozen: bool = True,
        freeze_dropout: bool = False,
        freeze_expression: bool = True,
        freeze_decoder_first_layer: bool = True,
        freeze_batchnorm_encoder: bool = True,
        freeze_batchnorm_decoder: bool = False,
        freeze_classifier: bool = True,
    ):
        device = _device_from_use_gpu(use_gpu)
        attr_dict, var_names, load_state_dict, _ = _get_loaded_data(
            reference_model, device=device
        )

        if inplace_subset_query_vars:
            logger.debug("Subsetting query vars to reference vars.")
            adata._inplace_subset_var(var_names)
        _validate_var_names(adata, var_names)

        registry = attr_dict.pop("registry_")
        if _MODEL_NAME_KEY in registry and registry[_MODEL_NAME_KEY] != cls.__name__:
            raise ValueError("It appears you are loading a model from a different class.")
        if _SETUP_ARGS_KEY not in registry:
            raise ValueError(
                "Saved model does not contain original setup inputs. "
                "Cannot load the original setup."
            )

        cls.setup_anndata(
            adata,
            source_registry=registry,
            extend_categories=True,
            allow_missing_labels=True,
            **registry[_SETUP_ARGS_KEY],
        )

        model = _initialize_model(cls, adata, registry, attr_dict, None)
        model.old_adata_manager = model.get_anndata_manager(adata, required=True)

        if REGISTRY_KEYS.CAT_COVS_KEY in model.adata_manager.data_registry:
            raise NotImplementedError(
                "scArches currently does not support models with extra categorical covariates."
            )

        version_split = model.registry[_constants._SCVI_VERSION_KEY].split(".")
        if int(version_split[1]) < 8 and int(version_split[0]) == 0:
            warnings.warn(
                "Query integration should be performed using models trained with version >= 0.8"
            )

        model.to_device(device)
        old_model = deepcopy(model)
        if hasattr(model.module, "register_ewc_buffers_from_state_dict"):
            restored = model.module.register_ewc_buffers_from_state_dict(load_state_dict)
            if restored:
                n_ewc = int(model.module.get_buffer("ewc_snap_count").item())
                logger.info(
                    "Registered %d EWC replay snapshot(s) before query load.", n_ewc
                )
        new_state_dict = model.module.state_dict()
        for key, load_ten in list(load_state_dict.items()):
            if key not in new_state_dict:
                continue
            new_ten = new_state_dict[key]
            if new_ten.size() == load_ten.size():
                continue
            dim_diff = new_ten.size()[-1] - load_ten.size()[-1]
            fixed_ten = torch.cat([load_ten, new_ten[..., -dim_diff:]], dim=-1)
            load_state_dict[key] = fixed_ten

        model.module.load_state_dict(load_state_dict, strict=False)
        old_model_batch_extend = deepcopy(model)
        model.module.eval()

        _set_params_online_update(
            model.module,
            unfrozen=unfrozen,
            freeze_decoder_first_layer=freeze_decoder_first_layer,
            freeze_batchnorm_encoder=freeze_batchnorm_encoder,
            freeze_batchnorm_decoder=freeze_batchnorm_decoder,
            freeze_dropout=freeze_dropout,
            freeze_expression=freeze_expression,
            freeze_classifier=freeze_classifier,
        )
        model.is_trained_ = False
        model._model_summary_string = ("{}, n_replay:{}, n_control:{}").format(
            model._model_summary_string,
            None,
            None,
        )

        replay_adata = adata.uns[replay_uns_key]
        replay_adata = model._validate_anndata(replay_adata)
        _ewc_bs = max(
            1,
            int(
                os.environ.get(
                    "SCVI_EWC_BATCH_SIZE",
                    os.environ.get("SCVI_TRAIN_BATCH_SIZE", "256"),
                )
            ),
        )
        rehearsal_rdl = model._make_data_loader(replay_adata, batch_size=_ewc_bs)

        if control_uns_key is not None:
            ctrl_adata = adata.uns[control_uns_key]
            ctrl_adata = old_model_batch_extend._validate_anndata(ctrl_adata)
            ctrl_rdl = old_model_batch_extend._make_data_loader(ctrl_adata, batch_size=_ewc_bs)
        else:
            ctrl_rdl = None

        model.module.importances = model._compute_importances(
            model=old_model, dataloader=rehearsal_rdl
        )
        if ctrl_rdl is not None:
            model.module.ctrl_importances = model._compute_importances(
                model=old_model_batch_extend, dataloader=ctrl_rdl
            )
        else:
            model.module.ctrl_importances = [
                (k, torch.zeros_like(p).to(p.device))
                for k, p in model.module.named_parameters()
                if p.requires_grad
            ]

        model.module.old_params = [
            (k, p.clone().detach())
            for k, p in model.module.named_parameters()
            if p.requires_grad
        ]
        model.module.register_ewc_snapshots()
        return model

    def _compute_importances(self, model, dataloader):
        importances = _zerolike_params_dict(model.module)
        params = filter(lambda p: p.requires_grad, model.module.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-3, eps=0.01, weight_decay=1e-6)
        device = model.device
        model.module.eval()

        for batch in dataloader:
            tensors = {key: val.to(device) for key, val in batch.items()}
            optimizer.zero_grad()
            inference_inputs = model.module._get_inference_input(tensors)
            inference_outputs = model.module.inference(**inference_inputs)
            generative_inputs = model.module._get_generative_input(tensors, inference_outputs)
            generative_outputs = model.module.generative(**generative_inputs)
            scvi_loss = model.module.loss(tensors, inference_outputs, generative_outputs)
            scvi_loss.loss.backward()
            param_dict = [(n, p) for n, p in model.module.named_parameters() if p.requires_grad]
            for (k1, p), (k2, imp) in zip(param_dict, importances):
                assert k1 == k2
                if p.grad is not None:
                    imp += p.grad.data.clone().pow(2)

        for _, imp in importances:
            imp /= float(len(dataloader))
        return importances


def _zerolike_params_dict(model):
    return [
        (k, torch.zeros_like(p).to(p.device))
        for k, p in model.named_parameters()
        if p.requires_grad
    ]
