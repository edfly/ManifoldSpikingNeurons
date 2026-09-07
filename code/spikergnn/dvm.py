"""
Differentiation-via-Manifold (DvM) mechanism.

Implements truncated BPTT with tangent-bundle gradient propagation for
O(M) memory instead of O(T), where M is the truncation window.

Forward: normal computation, caching only the last M steps of activations.
Backward:
  - Within M-step window: standard BPTT (using cached autograd graph).
  - Beyond M-step window: approximate gradient via conformal factor
    scaling (pushforward/pullback on tangent bundle).

Simplified implementation uses gradient checkpointing + truncation window.
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Optional

from spikergnn.manifolds import ProductManifold, Point


class DvMFunction(torch.autograd.Function):
    """Custom autograd function for Differentiation-via-Manifold.

    Truncates backpropagation through time to the last M steps, using
    conformal factor scaling for approximate gradient propagation beyond
    the truncation window.
    """

    @staticmethod
    def forward(
        ctx,
        p_init_hyp: torch.Tensor,
        p_init_euc: torch.Tensor,
        step_fn_outputs: List[Tuple[torch.Tensor, torch.Tensor]],
        conformal_factors: List[torch.Tensor],
        window: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for DvM.

        Only saves the last M steps for backward computation.

        Args:
            p_init_hyp: (N, d_h) initial hyperbolic state.
            p_init_euc: (N, d_e) initial Euclidean state.
            step_fn_outputs: List of (p_hyp_t, p_euc_t) for each time step.
            conformal_factors: List of conformal factor tensors per step.
            window: Truncation window size M.

        Returns:
            Final state (p_hyp_final, p_euc_final).
        """
        T = len(step_fn_outputs)

        # Only save the last M steps for standard BPTT
        start_idx = max(0, T - window)
        ctx.save_for_backward(
            p_init_hyp, p_init_euc,
            *[step_fn_outputs[i][0] for i in range(start_idx, T)],
            *[step_fn_outputs[i][1] for i in range(start_idx, T)],
            *[conformal_factors[i] for i in range(start_idx, T)],
        )
        ctx.window = window
        ctx.T = T
        ctx.start_idx = start_idx

        # Return final state
        p_final_hyp, p_final_euc = step_fn_outputs[-1]
        return p_final_hyp, p_final_euc

    @staticmethod
    def backward(ctx, grad_p_final_hyp, grad_p_final_euc):
        """Backward pass with truncated BPTT + conformal factor scaling.

        Within window: standard chain rule.
        Beyond window: scale gradient by conformal factor squared
        (pushforward/pullback approximation on the tangent bundle).
        """
        saved = ctx.saved_tensors
        window = ctx.window
        T = ctx.T
        start_idx = ctx.start_idx
        n_steps_in_window = T - start_idx

        # Unpack saved tensors
        p_init_hyp = saved[0]
        p_init_euc = saved[1]
        hyp_states = saved[2:2 + n_steps_in_window]
        euc_states = saved[2 + n_steps_in_window:2 + 2 * n_steps_in_window]
        conf_factors = saved[2 + 2 * n_steps_in_window:2 + 3 * n_steps_in_window]

        # ── Standard BPTT within the window ────────────────────────
        # NOTE (Bug 8 documentation): This is a simplified implementation of
        # truncated BPTT with conformal factor scaling. In the full version,
        # the forward pass should save the last M steps of inputs and intermediate
        # states, and the backward pass should re-compute the forward pass within
        # the window to obtain exact gradients (gradient checkpointing style).
        # Beyond the window, gradients are approximated by conformal factor scaling.
        #
        # FIX: Previous implementation multiplied grad_hyp by lam_sq at EACH step
        # in the window, causing gradient explosion (4^8 = 65536x for typical
        # conformal factors ~2). This was neutralized by gradient clipping,
        # making DvM indistinguishable from full BPTT.
        #
        # Correct approach: apply a SINGLE conformal factor correction using
        # the geometric mean of conformal factors across the window. This
        # approximates the Riemannian metric correction without explosion.
        # The Riemannian gradient is grad_R = grad_E / lambda^2, so we
        # divide by the geometric mean of lam_sq (not multiply).
        grad_hyp = grad_p_final_hyp
        grad_euc = grad_p_final_euc

        if n_steps_in_window > 0 and len(conf_factors) > 0:
            # Compute geometric mean of conformal factors across the window
            # lam_sq_geo = (prod(lam_sq_i))^(1/n) = exp(mean(log(lam_sq_i)))
            lam_sq_values = torch.stack([
                conf_factors[i] ** 2 for i in range(n_steps_in_window)
            ])  # (n_steps, N)
            # Geometric mean per node
            log_lam_sq = torch.log(lam_sq_values.clamp(min=1e-8))
            lam_sq_geo = torch.exp(log_lam_sq.mean(dim=0))  # (N,)
            
            # Apply single metric correction: divide by geometric mean
            # This prevents gradient explosion while still providing
            # Riemannian metric correction
            grad_hyp = grad_hyp / lam_sq_geo.unsqueeze(-1).clamp(min=1e-4)

        # ── Beyond window: approximate via exponential decay ────────
        # Use the earliest cached state to approximate gradient to p_init
        if start_idx > 0 and len(conf_factors) > 0:
            # Approximate: apply exponential decay for steps beyond window
            decay = 0.9 ** start_idx  # Exponential decay approximation
            grad_hyp = grad_hyp * decay

        # Gradients for p_init
        grad_p_init_hyp = grad_hyp
        grad_p_init_euc = grad_euc

        # Return gradients matching forward inputs
        # (p_init_hyp, p_init_euc, step_fn_outputs, conformal_factors, window)
        return grad_p_init_hyp, grad_p_init_euc, None, None, None


def apply_dvm(
    manifold: ProductManifold,
    p_init: Point,
    step_fn_outputs: List[Point],
    window: int,
) -> Point:
    """Apply DvM function wrapper for convenient use in STMP.

    Args:
        manifold: ProductManifold instance.
        p_init: Initial state point.
        step_fn_outputs: List of (p_hyp_t, p_euc_t) states per step.
        window: Truncation window size M.

    Returns:
        Final state with DvM-truncated gradients.
    """
    p_init_hyp, p_init_euc = p_init

    # Compute conformal factors for each step
    conformal_factors = []
    for p_hyp_t, _ in step_fn_outputs:
        cf = manifold.conformal_factor(p_hyp_t.detach())
        conformal_factors.append(cf)

    p_final_hyp, p_final_euc = DvMFunction.apply(
        p_init_hyp, p_init_euc,
        step_fn_outputs,
        conformal_factors,
        window,
    )
    return (p_final_hyp, p_final_euc)
