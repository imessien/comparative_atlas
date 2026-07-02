from ._scanvae import SCANVAE

from typing import Literal

from scvi.nn import FCLayers
from collections import namedtuple

AdaptHeadLoss = namedtuple(
    "AdaptHeadLoss",
    ["loss", "reconstruction_loss", "energy_score_loss"],
    defaults=(None, None, None),
)


from scvi.nn import DecoderSCVI
from scvi.distributions import NegativeBinomial
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl
import torch
import numpy as np
from torch.linalg import vector_norm

class Adapt(SCANVAE):
    """Align observational SCANVI latents on Parse (or other query) gene counts.

    **Parse pipeline (``energy_only=True``)** — primary path in ``perturbation.py``:

    * **Target** ``embedding``: reference SCANVI latent for each Parse PBS adapt cell
      (from ``ensure_energy_embedding_backed``, not scGPT or other obsm).
    * **Input** ``X``: backed Parse gene counts streamed via ``BackedScanviDataModule``.
    * **Loss**: energy score between target latent and ``m0`` encoder ``z`` on Parse counts.

    When ``energy_only`` is false, a projection head and NB panel decoder support
    legacy cross-modal adaptation (external embeddings with different dimension).
    """

    def __init__(
        self,
        m0_module: SCANVAE,
        n_input: int = 100,
        n_output: int = 100,
        n_hidden: int = 128,
        n_layers: int = 2,
        dropout_rate: float = 0.1,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        energy_only: bool | None = None,
        **model_kwargs,
    ):
        model_kwargs = dict(model_kwargs)
        n_latent = getattr(m0_module, "n_latent", 10)
        if energy_only is None:
            energy_only = n_input == n_latent
        self.energy_only = bool(energy_only)
        model_kwargs.setdefault("n_batch", getattr(m0_module, "n_batch", 0))
        model_kwargs.setdefault("n_labels", max(1, getattr(m0_module, "n_labels", 1)))
        model_kwargs.setdefault("n_latent", n_latent)
        model_kwargs.setdefault("dispersion", getattr(m0_module, "dispersion", "gene"))
        model_kwargs.setdefault(
            "gene_likelihood", getattr(m0_module, "gene_likelihood", "zinb")
        )

        super().__init__(
            n_input=n_input,
            n_hidden=n_hidden,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            **model_kwargs,
        )
        if self.energy_only:
            self.projection_layer = None
            self.decoder = None
            self.px_r_m0 = None
        else:
            self.projection_layer = FCLayers(
                n_in=n_input,
                n_out=n_latent,
                n_layers=n_layers,
                n_hidden=n_hidden,
                dropout_rate=dropout_rate,
                use_batch_norm=use_batch_norm,
                use_layer_norm=use_layer_norm,
            )
            self.decoder = DecoderSCVI(
                n_input=n_latent,
                n_output=n_output,
                n_layers=n_layers,
                n_hidden=n_hidden,
                use_batch_norm=use_batch_norm,
                use_layer_norm=use_layer_norm,
            )
            self.px_r_m0 = torch.nn.Parameter(torch.randn(n_output))
        self.init_params_ = self._get_init_params(locals())
        self.was_pretrained = False
        self.m0 = m0_module
        self.use_embedding_for_inference = True

        # EWC state. Empty until `register_ewc_anchor` is called; while empty the
        # penalty computed in `SCANVAE.loss_with_replay` is 0 (the zip is empty).
        self.old_params = []
        self.importances = []
        self.ctrl_importances = []

    def _get_init_params(self, locals):
        return {k: v for k, v in locals.items() if k != "self"}

    def _get_inference_input(self, tensors):
        """Route gene-count batches through ``m0`` when not in embedding mode."""
        if not self.use_embedding_for_inference:
            return self.m0._get_inference_input(tensors)
        inputs = super()._get_inference_input(tensors)
        if "embedding" in tensors:
            inputs["x"] = tensors["embedding"]
        return inputs

    def inference(self, *args, **kwargs):
        if not self.use_embedding_for_inference:
            return self.m0.inference(*args, **kwargs)
        return super().inference(*args, **kwargs)

    def _get_generative_input(self, tensors, inference_outputs, **kwargs):
        if not self.use_embedding_for_inference:
            return self.m0._get_generative_input(tensors, inference_outputs, **kwargs)
        return super()._get_generative_input(tensors, inference_outputs, **kwargs)

    def generative(self, *args, **kwargs):
        if not self.use_embedding_for_inference:
            return self.m0.generative(*args, **kwargs)
        return super().generative(*args, **kwargs)

    def _source_latent(self, emb: torch.Tensor) -> torch.Tensor:
        if self.projection_layer is not None:
            return self.projection_layer(emb)
        return emb

    def _adaptation_head_loss(self, tensors, m0_inference_outputs):
        """Energy alignment of source latent to ``m0`` gene-count encoder ``z``."""
        if "embedding" not in tensors:
            raise KeyError(
                "Adaptation loss expected `embedding` in tensors (reference SCANVI latent). "
                f"Available keys: {list(tensors.keys())}"
            )
        emb = tensors["embedding"]
        z_target = self._source_latent(emb)
        m0_z = m0_inference_outputs["z"]
        if z_target.shape[-1] != m0_z.shape[-1]:
            raise ValueError(
                "Latent dim mismatch for energy TTA: "
                f"target embedding {tuple(z_target.shape)} vs m0 z {tuple(m0_z.shape)}. "
                "Parse PBS adapt expects reference SCANVI latents (same n_latent as m0), "
                "not an external embedding (e.g. scGPT) unless energy_only=False with a projection head."
            )
        if not any(p.requires_grad for p in self.m0.parameters()):
            m0_z = m0_z.detach()
        energy_score_loss = self.energy_loss(z_target, m0_z, verbose=False)

        if self.energy_only:
            loss = energy_score_loss.mean()
            zero = torch.zeros((), device=loss.device, dtype=loss.dtype)
            return AdaptHeadLoss(
                loss=loss,
                reconstruction_loss=zero,
                energy_score_loss=energy_score_loss.mean(),
            )

        if "x_m1" not in tensors:
            raise KeyError(
                "Panel adaptation expected `x_m1` in tensors. "
                f"Available keys: {list(tensors.keys())}"
            )
        x_m1 = tensors["x_m1"]
        library_emb = torch.log(x_m1.sum(dim=1, keepdim=True).clamp_min(1e-8))
        _, _, px_rate_emb, _ = self.decoder("gene", z_target, library_emb)
        theta = torch.exp(self.px_r_m0.clamp(min=-12, max=12))
        if theta.shape[-1] != px_rate_emb.shape[-1]:
            raise ValueError(
                "Adapt decoder output and px_r_m0 must match X_target gene "
                f"dimension (got px_rate {px_rate_emb.shape[-1]} vs "
                f"px_r_m0 {theta.shape[-1]})."
            )
        reconst_loss_emb = -NegativeBinomial(mu=px_rate_emb, theta=theta).log_prob(
            x_m1
        ).sum(dim=-1)
        loss = (reconst_loss_emb + energy_score_loss).mean()
        return AdaptHeadLoss(
            loss=loss,
            reconstruction_loss=reconst_loss_emb.mean(),
            energy_score_loss=energy_score_loss.mean(),
        )

    def _replay_forward(self, *args, **kwargs):
        raise NotImplementedError(
            "Adapt uses TTA_SCANVI.train_test_time_adaptation(); "
            "SCANVI Lightning replay is not supported."
        )

    def register_ewc_anchor(self, importances=None, ctrl_importances=None):
        """Snapshot the current trainable params as the EWC anchor.

        After calling this, `SCANVAE.loss_with_replay` regularizes the module's
        trainable parameters toward this snapshot, weighted by the (Fisher)
        importances. This anchors the *adaptation* module to its current state;
        it does not touch or reference the reference ``m0`` weights.

        Parameters
        ----------
        importances
            List of ``(name, tensor)`` importances aligned with the module's
            trainable ``named_parameters()`` (e.g. produced by
            ``ADAPT._compute_importances``). If ``None``, uniform importances
            (ones) are used, i.e. a plain quadratic anchor.
        ctrl_importances
            List of ``(name, tensor)`` control importances. If ``None``, ones
            are used so the ``"product"`` penalty reduces to
            ``importance * (param - anchor) ** 2``.
        """
        self.old_params = [
            (n, p.clone().detach())
            for n, p in self.named_parameters()
            if p.requires_grad
        ]
        if importances is None:
            importances = [(n, torch.ones_like(p)) for n, p in self.old_params]
        if ctrl_importances is None:
            ctrl_importances = [(n, torch.ones_like(p)) for n, p in self.old_params]
        self.importances = importances
        self.ctrl_importances = ctrl_importances
    
    def vectorize(self,x, multichannel=False):
        """Vectorize data in any shape.

        Args:
            x (torch.Tensor): input data
            multichannel (bool, optional): whether to keep the multiple channels (in the second dimension). Defaults to False.

        Returns:
            torch.Tensor: data of shape (sample_size, dimension) or (sample_size, num_channel, dimension) if multichannel is True.
        """
        if len(x.shape) == 1:
            return x.unsqueeze(1)
        if len(x.shape) == 2:
            return x
        else:
            if not multichannel: # one channel
                return x.reshape(x.shape[0], -1)
            else: # multi-channel
                return x.reshape(x.shape[0], x.shape[1], -1)

    def energy_loss(self, x_true, x_est, beta=1, verbose=True):
        """
        Energy score loss, returned per data example (not averaged).

        Args:
            x_true (torch.Tensor): shape [N, D]
            x_est (list of Tensors or a single tensor): 
                - List of M tensors of shape [N, D], or 
                - Tensor of shape [N*M, D] to be split into M samples.
            beta (float): power parameter.
            verbose (bool): if True, also return s1 and s2 terms per example.

        Returns:
            Tensor of shape [N] (if verbose=False), or (loss, s1, s2) if verbose=True.
        """
        if isinstance(beta, torch.Tensor):
            beta_val = beta.item()
        else:
            beta_val = float(beta)
        EPS = 0 if beta_val.is_integer() else 1e-5
        x_true = self.vectorize(x_true).unsqueeze(1)  # shape: [N, 1, D]

        if not isinstance(x_est, list):
            N = x_true.shape[0]
            M = x_est.shape[0] // N
            x_est = list(torch.split(x_est, N, dim=0))
        M = len(x_est)
        x_est = [self.vectorize(xi).unsqueeze(1) for xi in x_est]  # each: [N, 1, D]
        x_est = torch.cat(x_est, dim=1)  # shape: [N, M, D]

        # --- s1: distance from x_true to each sample ---
        s1 = (vector_norm(x_est - x_true, 2, dim=2) + EPS).pow(beta).mean(dim=1)  # shape: [N]

        # --- s2: average pairwise distance among samples per example ---
        # For M <= 1, the pairwise term is undefined (division by zero in
        # unbiased scaling), so we set it to 0.
        if M <= 1:
            s2 = torch.zeros_like(s1)
        else:
            dists = torch.cdist(x_est, x_est, p=2) + EPS  # shape: [N, M, M]
            s2 = dists.pow(beta).mean(dim=(1, 2)) * M / (M - 1)  # shape: [N]

        # --- final loss per example ---
        loss = s1 - s2 / 2  # shape: [N]

        
        if verbose:
            return loss, s1, s2
        else:
            return loss



    def energy_loss_two_sample(self, x0, x, xp, x0p=None, beta=1, verbose=True, weights=None, mask=None):
        """
        Per-example loss function based on the energy score (estimated from two samples).

        Args:
            x0 (torch.Tensor): Sample from the true distribution. Shape: [N, D]
            x (torch.Tensor): Sample from the estimated distribution. Shape: [N, D]
            xp (torch.Tensor): Another sample from the estimated distribution. Shape: [N, D]
            x0p (torch.Tensor, optional): Another sample from the true distribution. Shape: [N, D]
            beta (float): Power parameter in the energy score.
            verbose (bool): Whether to return s1, s2 (and s3 if x0p is given) per example.
            weights (float or torch.Tensor, optional): Scalar or tensor of shape [N] for per-example weights.

        Returns:
            If verbose:
                Tuple of three or four tensors of shape [N]: (loss, s1, s2[, s3])
            Else:
                Tensor of shape [N]: per-example loss
        """
        if isinstance(beta, torch.Tensor):
            beta_val = beta.item()
        else:
            beta_val = float(beta)
        EPS = 0 if beta_val.is_integer() else 1e-5

        x0 = self.vectorize(x0)
        x = self.vectorize(x)
        xp = self.vectorize(xp)

        if weights is None:
            weights = 1.0
        
        weights = torch.tensor(weights, device=x.device, dtype=x.dtype)
        if weights.ndim == 0:
            weights = weights.expand(x.shape[0])
        elif weights.ndim != 1 or weights.shape[0] != x.shape[0]:
            raise ValueError(f"Weights must be a scalar or a tensor of shape [{x.shape[0]}]")

        if x0p is None:
            # s1 terms
            s1_term1 = (vector_norm(x - x0, 2, dim=1) + EPS).pow(beta) / 2
            s1_term2 = (vector_norm(xp - x0, 2, dim=1) + EPS).pow(beta) / 2
            s1 = s1_term1 + s1_term2

            # s2 term
            s2 = (vector_norm(x - xp, 2, dim=1) + EPS).pow(beta) / 2

            loss = (s1 - s2) * weights

            if mask is not None:
                if not torch.is_floating_point(mask) and mask.dtype != torch.bool:
                    mask = mask.bool()
                loss = loss[mask]
                s1 = s1[mask]
                s2 = s2[mask]
                weights = weights[mask]
            if verbose:
                return loss, s1 * weights, s2 * weights
            else:
                return loss

        else:
            x0p = self.vectorize(x0p)

            # s1 terms
            s1_term1 = (vector_norm(x - x0, 2, dim=1) + EPS).pow(beta) / 4
            s1_term2 = (vector_norm(xp - x0, 2, dim=1) + EPS).pow(beta) / 4
            s1_term3 = (vector_norm(x - x0p, 2, dim=1) + EPS).pow(beta) / 4
            s1_term4 = (vector_norm(xp - x0p, 2, dim=1) + EPS).pow(beta) / 4
            s1 = s1_term1 + s1_term2 + s1_term3 + s1_term4

            # s2 and s3 terms
            s2 = (vector_norm(x - xp, 2, dim=1) + EPS).pow(beta) / 2
            s3 = (vector_norm(x0 - x0p, 2, dim=1) + EPS).pow(beta) / 2

            loss = (s1 - s2 - s3) * weights

            if mask is not None:
                if not torch.is_floating_point(mask) and mask.dtype != torch.bool:
                    mask = mask.bool()
                loss = loss[mask]
                s1 = s1[mask]
                s2 = s2[mask]
                weights = weights[mask]
            if verbose:
                return loss, s1 * weights, s2 * weights, s3 * weights
            else:
                return loss


    