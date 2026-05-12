"""Student starter template for the Spectral Optimizer Capstone (DS 6210).

This file is YOURS to edit.  Do not edit anything in `harness/`.

Three modes (§2 of the handout):

    disabled   -- exactly AdamW (already implemented; do not break parity)
    debug      -- applies AdamW but logs spectral diagnostics each step
    production -- YOUR custom update rule (raises NotImplementedError until
                  you fill in `_compute_spectral_correction`)

A pure-Python AdamW reference is built into the disabled/debug branches
so that `parity_check.py` can prove your starter matches torch.optim.AdamW
to floating-point tolerance.  Once you fill in `production` you must
keep parity for `disabled` -- the parity test is the only behavioral
guard the grader runs against the harness.

A safe trajectory template (from §2 of the handout) is sketched in
`_compute_spectral_correction`.  You are free to use it, replace it,
or invent something entirely different -- as long as the optimizer is
honestly per-tensor and respects the rules in `docs/problem.tex` §1.
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Literal

import torch


Mode = Literal["disabled", "debug", "production"]


class SpectralOptimizer(torch.optim.Optimizer):
    """A 3-mode skeleton for a per-tensor low-rank spectral optimizer.

    Args:
        params: parameters or parameter groups (same shape as torch.optim.AdamW).
        lr, betas, eps, weight_decay: standard AdamW hyperparameters.
        mode: one of {"disabled", "debug", "production"}.
        spectral_rank: per-tensor rank of the spectral subspace (default 8).
            Used only by `debug` and `production`.  Must be <= min(weight.shape).
        spectral_beta: EMA decay for whatever moment you maintain in the
            spectral subspace.  The starter does not consume this; it is
            here so your `production` branch has a hyperparameter to tune.
        correction_strength: starter scalar gate on your spectral
            correction.  `disabled`/`debug` ignore it.

    The optimizer maintains, for each 2-D parameter:
        state["step"]   : int
        state["m"]      : first moment, shape == p.shape
        state["v"]      : second moment, shape == p.shape
        state["basis"]  : optional per-tensor orthonormal frame
                          (shape (r, d_flat)), allocated lazily on first
                          `debug`/`production` step.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        mode: Mode = "disabled",
        spectral_rank: int = 8,
        spectral_beta: float = 0.95,
        correction_strength: float = 0.03,
    ) -> None:
        if mode not in ("disabled", "debug", "production"):
            raise ValueError(
                f"mode must be one of disabled/debug/production, got {mode!r}",
            )
        if not 0.0 <= lr:
            raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")
        if spectral_rank < 1:
            raise ValueError(f"spectral_rank must be >= 1, got {spectral_rank}")

        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            mode=mode, spectral_rank=spectral_rank,
            spectral_beta=spectral_beta,
            correction_strength=correction_strength,
        )
        super().__init__(params, defaults)
        # Per-step diagnostics from `debug` mode are appended here for
        # student inspection.  Cleared on every `step()` call.
        self.last_diagnostics: dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Per-step diagnostics get aggregated across all tensors and
        # then averaged before being stored on `last_diagnostics`.
        diag_acc = {"effective_rank": 0.0, "cosine_with_adamw": 0.0,
                    "spectral_correction_norm": 0.0, "n_tensors": 0}

        for group in self.param_groups:
            mode = group["mode"]

            # Exact PyTorch AdamW path for disabled mode.
            if mode == "disabled":
                params_with_grad = []
                grads = []
                exp_avgs = []
                exp_avg_sqs = []
                max_exp_avg_sqs = []
                state_steps = []

                for p in group["params"]:
                    if p.grad is None:
                        continue

                    state = self.state[p]

                    if len(state) == 0:
                        state["step"] = torch.zeros(())
                        state["m"] = torch.zeros_like(p)
                        state["v"] = torch.zeros_like(p)

                    params_with_grad.append(p)
                    grads.append(p.grad)
                    exp_avgs.append(state["m"])
                    exp_avg_sqs.append(state["v"])
                    state_steps.append(state["step"])

                if len(params_with_grad) > 0:
                    beta1, beta2 = group["betas"]

                    torch.optim._functional.adamw(
                        params_with_grad,
                        grads,
                        exp_avgs,
                        exp_avg_sqs,
                        max_exp_avg_sqs,
                        state_steps,
                        foreach=True,
                        capturable=False,
                        differentiable=False,
                        fused=False,
                        grad_scale=None,
                        found_inf=None,
                        has_complex=False,
                        amsgrad=False,
                        beta1=beta1,
                        beta2=beta2,
                        lr=group["lr"],
                        weight_decay=group["weight_decay"],
                        eps=group["eps"],
                        maximize=False,
                    )

                continue

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                # AdamW direction is needed by all three modes.
                d_t, m_hat, v_hat = _adamw_direction(p, state, group)

                if mode == "debug":
                    update = d_t
                    if p.dim() == 2:
                        diag = self._diagnostics_only(p, d_t, group, state)
                        for k in ("effective_rank", "cosine_with_adamw",
                                  "spectral_correction_norm"):
                            diag_acc[k] += diag.get(k, 0.0)
                        diag_acc["n_tensors"] += 1
                else:  # production
                    update = d_t
                    if p.dim() == 2:
                        correction = self._compute_spectral_correction(
                            p, d_t, group, state,
                        )
                        update = update + correction

                        diag = state.get("last_diag")
                        if diag is not None:
                            for k in ("effective_rank", "cosine_with_adamw",
                                      "spectral_correction_norm"):
                                diag_acc[k] += diag.get(k, 0.0)
                            diag_acc["n_tensors"] += 1

                lr = group["lr"]
                wd = group["weight_decay"]
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(update, alpha=-lr)

        if diag_acc["n_tensors"] > 0:
            n = diag_acc.pop("n_tensors")
            self.last_diagnostics = {k: v / n for k, v in diag_acc.items()}
        else:
            self.last_diagnostics = {}

        return loss

    # ------------------------------------------------------------------ #
    # Student extension point
    # ------------------------------------------------------------------ #

    def _compute_spectral_correction(
        self,
        p: torch.nn.Parameter,
        d_t: torch.Tensor,
        group: dict,
        state: dict,
    ) -> torch.Tensor:
        """Return the additive spectral correction to the AdamW update.

        Final update = AdamW direction + this.

        The starter raises so that `production` mode is opt-in.  Replace
        the body with your own implementation.

        A safe template (§2 of the handout): per-tensor low-rank angular
        steering.  For a 2-D weight ``p`` with AdamW direction ``d_t``,
        let ``x_t = d_t / ||d_t||`` and maintain ``B_t in R^{r x d}`` with
        ``B_t B_t.T = I``.  Compute coefficients ``c_t = B_t @ x_t``,
        choose ``Phi_t in R^r``, and emit

            y_t = (x_t + B_t.T @ Phi_t) / ||x_t + B_t.T @ Phi_t||
            correction = ||d_t|| * y_t - d_t

        so that ``d_t + correction = ||d_t|| * y_t`` -- AdamW step
        magnitude preserved, only direction rotated.

        See `_starter_low_rank_template` below for the scaffolding you
        can cut/paste/edit.  It is intentionally NOT wired in by default
        because (a) you should think about it before you use it and
        (b) any starter spectral choice would prejudge your hypothesis.
        """
        # Only apply spectral corrections to 2-D tensors.
        if p.dim() != 2:
            return torch.zeros_like(d_t)

        with torch.no_grad():

            # -----------------------------
            # Hyperparameters
            # -----------------------------
            k = int(group["spectral_rank"])
            strength = float(group["correction_strength"])
            warmup_steps = 10

            # -----------------------------
            # Flatten AdamW direction
            # -----------------------------
            d_flat = d_t.reshape(-1)

            norm_d = d_flat.norm()
            if norm_d.item() == 0.0:
                return torch.zeros_like(d_t)

            x_t = d_flat / (norm_d + 1e-12)

            # -----------------------------
            # Initialize trajectory buffer
            # -----------------------------
            if "dir_buffer" not in state:
                state["dir_buffer"] = []
                state["traj_step"] = 0

            buffer = state["dir_buffer"]

            state["traj_step"] += 1
            traj_step = state["traj_step"]

            # -----------------------------
            # Warmup phase
            # -----------------------------
            if len(buffer) == 0 or traj_step <= warmup_steps:

                buffer.append(x_t.detach().clone())

                if len(buffer) > k:
                    buffer.pop(0)

                state["last_diag"] = {
                    "effective_rank": 1.0,
                    "cosine_with_adamw": 1.0,
                    "spectral_correction_norm": 0.0,
                }

                return torch.zeros_like(d_t)

            # -----------------------------
            # Persistent trajectory direction
            # -----------------------------
            B = torch.stack(buffer, dim=0)

            phi = B.mean(dim=0)

            phi_norm = phi.norm()

            if phi_norm.item() == 0.0:

                buffer.append(x_t.detach().clone())

                if len(buffer) > k:
                    buffer.pop(0)

                state["last_diag"] = {
                    "effective_rank": 1.0,
                    "cosine_with_adamw": 1.0,
                    "spectral_correction_norm": 0.0,
                }

                return torch.zeros_like(d_t)

            phi = phi / (phi_norm + 1e-12)

            # -----------------------------
            # Norm-preserving angular correction
            # -----------------------------
            raw = x_t + strength * phi

            y_t = raw / (raw.norm() + 1e-12)

            correction_flat = norm_d * y_t - d_flat

            correction = correction_flat.reshape_as(d_t)

            # -----------------------------
            # Diagnostics
            # -----------------------------
            cosine = torch.dot(x_t, y_t).item()

            correction_norm = correction_flat.norm().item()

            projected_energy = torch.dot(x_t, phi).pow(2).item()

            state["last_diag"] = {
                "effective_rank": projected_energy * k,
                "cosine_with_adamw": cosine,
                "spectral_correction_norm": correction_norm,
            }

            # -----------------------------
            # Update trajectory buffer
            # -----------------------------
            buffer.append(x_t.detach().clone())

            if len(buffer) > k:
                buffer.pop(0)

            return correction

    # ------------------------------------------------------------------ #
    # Diagnostics  (used by debug mode and your figures)
    # ------------------------------------------------------------------ #

    def _diagnostics_only(
        self,
        p: torch.nn.Parameter,
        d_t: torch.Tensor,
        group: dict,
        state: dict,
    ) -> dict[str, float]:
        """Compute spectral diagnostics for `p` WITHOUT mutating its update.

        Returns a dict with at least:
            effective_rank       : entropy-based rank of the gradient
                                   covariance proxy (1 .. spectral_rank).
            cosine_with_adamw    : <correction, d_t> / (||correction|| * ||d_t||)
                                   for whatever your trial correction would be.
                                   The starter computes a placeholder of 1.0 since
                                   it has no live correction yet.
            spectral_correction_norm
                                 : Frobenius norm of the would-be correction;
                                   0.0 in the starter for the same reason.
        """
        with torch.no_grad():
            d_flat = d_t.reshape(-1)
            x_t = d_flat / (d_flat.norm() + 1e-12)
            r = min(group["spectral_rank"], p.numel())
            basis = state.get("basis")
            if basis is None or basis.shape != (r, p.numel()):
                basis = _orthonormal_rows(r, p.numel(), device=p.device,
                                          dtype=p.dtype, generator_seed=0)
                state["basis"] = basis
            c_t = basis @ x_t
            energy = c_t.pow(2).sum().clamp(min=1e-12, max=1.0)
            # Effective rank by entropy of normalized squared coefficients.
            probs = c_t.pow(2) / c_t.pow(2).sum().clamp(min=1e-12)
            entropy = -(probs * (probs.clamp_min(1e-12)).log()).sum()
            eff_rank = float(entropy.exp().item())
            return {
                "effective_rank": eff_rank,
                "cosine_with_adamw": 1.0,
                "spectral_correction_norm": 0.0,
            }


# ---------------------------------------------------------------------- #
# Free functions: AdamW reference + scaffolding helpers.
# ---------------------------------------------------------------------- #


def _adamw_direction(
    p: torch.nn.Parameter,
    state: dict,
    group: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the AdamW direction d_t = m_hat / (sqrt(v_hat) + eps).

    Updates state["m"], state["v"], state["step"] in place.  Decoupled
    weight decay is applied separately by the caller (it is a
    multiplicative shrink on theta, not on the direction).

    Returns ``(d_t, m_hat, v_hat)``.  d_t has the same shape as ``p``.
    """
    grad = p.grad
    beta1, beta2 = group["betas"]
    eps = group["eps"]

    state["step"] += 1
    t = state["step"]
    state["m"].mul_(beta1).add_(grad, alpha=1.0 - beta1)
    state["v"].mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

    bias1 = 1.0 - beta1 ** t
    bias2 = 1.0 - beta2 ** t
    m_hat = state["m"] / bias1
    v_hat = state["v"] / bias2

    d_t = m_hat / (v_hat.sqrt() + eps)
    return d_t, m_hat, v_hat


def _orthonormal_rows(
    r: int, d: int,
    *, device, dtype, generator_seed: int = 0,
) -> torch.Tensor:
    """Return a frozen orthonormal frame B in R^{r x d}, B @ B.T = I_r.

    Uses a CPU torch.Generator for reproducibility, then moves to GPU.
    """
    g = torch.Generator(device="cpu").manual_seed(generator_seed)
    raw = torch.randn(r, d, generator=g)
    # QR on a tall matrix (d >= r); take the Q rows.
    q, _ = torch.linalg.qr(raw.T, mode="reduced")
    return q.T.to(device=device, dtype=dtype).contiguous()


def _starter_low_rank_template(
    d_t: torch.Tensor,
    basis: torch.Tensor,
    phi: torch.Tensor,
) -> torch.Tensor:
    """Reference implementation of the §2 angular-correction template.

    Given AdamW direction ``d_t`` (shape == weight), an orthonormal
    frame ``basis`` of shape (r, d_flat), and a coefficient vector
    ``phi`` of shape (r,) chosen by the student, returns the additive
    correction such that

        d_t + correction = ||d_t|| * y_t,    y_t = (x_t + B.T @ phi) / ||...||

    with ``x_t = d_t / ||d_t||``.  Step magnitude is preserved exactly.

    Cut and paste this into your `_compute_spectral_correction` once you
    have decided how to build ``phi`` from the trajectory state.
    """
    d_flat = d_t.reshape(-1)
    norm = d_flat.norm() + 1e-12
    x_t = d_flat / norm
    raw = x_t + basis.T @ phi
    y_t = raw / (raw.norm() + 1e-12)
    correction_flat = norm * y_t - d_flat
    return correction_flat.reshape_as(d_t)
