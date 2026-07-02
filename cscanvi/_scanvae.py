from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import torch
from scvi.module import SCANVAE as _SCANVAE
from scvi.module.base import LossOutput, auto_move_data

if TYPE_CHECKING:
    from collections.abc import Sequence

# Keys forwarded to ``SCANVAE.loss``; plan/optimizer kwargs (e.g. ``m0_lr``) are stripped.
_SCANVAE_LOSS_KEYS = frozenset(
    {
        "feed_labels",
        "kl_weight",
        "labelled_tensors",
        "classification_ratio",
        "replay",
    }
)


class SCANVAE(_SCANVAE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.combine_type = "product"
        self.old_params: list = []

    def register_ewc_snapshots(self) -> None:
        importances = getattr(self, "importances", None)
        ctrl_importances = getattr(self, "ctrl_importances", None)
        old_params = getattr(self, "old_params", None)
        if not old_params or importances is None or ctrl_importances is None:
            return
        for name in list(self._buffers.keys()):
            if name.startswith("ewc_snap_"):
                del self._buffers[name]
        for i, ((kn, to), (ki, ti), (kc, tc)) in enumerate(
            zip(old_params, importances, ctrl_importances)
        ):
            if kn != ki or kn != kc:
                raise ValueError(
                    f"EWC tensor lists misaligned at index {i}: {kn!r}, {ki!r}, {kc!r}"
                )
            self.register_buffer(f"ewc_snap_old_{i}", to.detach().clone())
            self.register_buffer(f"ewc_snap_imp_{i}", ti.detach().clone())
            self.register_buffer(f"ewc_snap_ctrl_{i}", tc.detach().clone())
        n = len(old_params)
        self.register_buffer("ewc_snap_count", torch.tensor([n], dtype=torch.long))

    def register_ewc_buffers_from_state_dict(self, state_dict: dict) -> bool:
        ewc_keys = [k for k in state_dict if k.startswith("ewc_snap_")]
        if not ewc_keys:
            return False
        for name in list(self._buffers.keys()):
            if name.startswith("ewc_snap_"):
                del self._buffers[name]
        for key in ewc_keys:
            self.register_buffer(key, state_dict[key].detach().clone())
        return True

    def _ensure_ewc_lists_from_buffers(self) -> None:
        if getattr(self, "old_params", None) and len(self.old_params) > 0:
            return
        if "ewc_snap_count" not in self._buffers:
            return
        n = int(self.get_buffer("ewc_snap_count").item())
        req_names = [name for name, p in self.named_parameters() if p.requires_grad]
        if len(req_names) != n or n == 0:
            return
        old_out = []
        imp_out = []
        ctrl_out = []
        for i, name in enumerate(req_names):
            old_out.append((name, self.get_buffer(f"ewc_snap_old_{i}")))
            imp_out.append((name, self.get_buffer(f"ewc_snap_imp_{i}")))
            ctrl_out.append((name, self.get_buffer(f"ewc_snap_ctrl_{i}")))
        self.old_params = old_out
        self.importances = imp_out
        self.ctrl_importances = ctrl_out

    def loss_with_replay(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        loss_kwargs=None,
    ) -> LossOutput:
        loss_kwargs = dict(loss_kwargs or {})
        ewc_importance = float(loss_kwargs.pop("ewc_importance", 0.0))
        scanvi_loss_kwargs = {
            key: loss_kwargs[key] for key in _SCANVAE_LOSS_KEYS if key in loss_kwargs
        }
        loss_output = self.loss(
            tensors, inference_outputs, generative_outputs, **scanvi_loss_kwargs
        )
        self._ensure_ewc_lists_from_buffers()

        old_params = getattr(self, "old_params", None)
        if (
            ewc_importance == 0
            or old_params is None
            or len(old_params) == 0
            or not hasattr(self, "importances")
            or not hasattr(self, "ctrl_importances")
        ):
            return loss_output

        old_by_name = dict(old_params)
        imp_by_name = dict(self.importances)
        ctrl_by_name = dict(self.ctrl_importances)
        penalty = torch.tensor(0.0, device=loss_output.loss.device)

        for name, cur_param in self.named_parameters():
            if name not in old_by_name:
                continue
            dev = cur_param.device
            saved_param = old_by_name[name].to(dev)
            imp = imp_by_name[name].to(dev)
            ctrl_imp = ctrl_by_name[name].to(dev)
            if cur_param.size() == saved_param.size():
                if self.combine_type == "product":
                    penalty += (
                        (imp * ctrl_imp) * (cur_param - saved_param).pow(2)
                    ).sum()
                if self.combine_type == "additive":
                    penalty += (
                        (imp + ctrl_imp) * (cur_param - saved_param).pow(2)
                    ).sum()

        return replace(loss_output, loss=loss_output.loss + ewc_importance * penalty)

    @auto_move_data
    def _replay_forward(
        self,
        tensors,
        get_inference_input_kwargs: dict | None = None,
        get_generative_input_kwargs: dict | None = None,
        inference_kwargs: dict | None = None,
        generative_kwargs: dict | None = None,
        loss_kwargs: dict | None = None,
        compute_loss: bool = True,
    ):
        get_inference_input_kwargs = get_inference_input_kwargs or {}
        get_generative_input_kwargs = get_generative_input_kwargs or {}
        inference_kwargs = inference_kwargs or {}
        generative_kwargs = generative_kwargs or {}
        loss_kwargs = loss_kwargs or {}

        inference_inputs = self._get_inference_input(tensors, **get_inference_input_kwargs)
        inference_outputs = self.inference(**inference_inputs, **inference_kwargs)
        generative_inputs = self._get_generative_input(
            tensors, inference_outputs, **get_generative_input_kwargs
        )
        generative_outputs = self.generative(**generative_inputs, **generative_kwargs)

        if compute_loss:
            losses = self.loss_with_replay(
                tensors, inference_outputs, generative_outputs, loss_kwargs
            )
            return inference_outputs, generative_outputs, losses
        return inference_outputs, generative_outputs
