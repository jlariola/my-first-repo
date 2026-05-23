"""PyTorch implementation of HN_Adam (Algorithm 2)."""

from __future__ import annotations

import random
from typing import Callable, Iterable, Optional

import torch
from torch import Tensor
from torch.optim import Optimizer


class HNAdam(Optimizer):
    """Hybrid and adaptive norm Adam optimizer (HN_Adam)."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.99),
        eps: float = 1e-8,
        lambda_t0: Optional[float] = None,
    ) -> None:
        if params is None:
            raise ValueError("params cannot be None.")
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if len(betas) != 2:
            raise ValueError("betas must be a tuple of two floats")

        beta1, beta2 = betas
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta1 value: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2 value: {beta2}")

        # Lambda_t0 is randomly chosen in the range [2, 4].
        if lambda_t0 is None:
            lambda_t0 = random.uniform(2.0, 4.0)
        if not 2.0 <= lambda_t0 <= 4.0:
            raise ValueError(f"lambda_t0 must be in [2, 4], got {lambda_t0}")

        defaults = {
            "lr": lr,
            "betas": (beta1, beta2),
            "eps": eps,
            "lambda_t0": lambda_t0,
            "amsgrad": False,
        }
        super().__init__(params, defaults)

        if len(self.param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], Tensor]] = None) -> Optional[Tensor]:
        loss: Optional[Tensor] = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Step 3: For all t = 1, ..., T do
        for group in self.param_groups:
            lr: float = group["lr"]
            beta1, beta2 = group["betas"]
            eps: float = group["eps"]
            lambda_t0: float = group["lambda_t0"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("HNAdam does not support sparse gradients")

                state = self.state[param]

                # Initialize: m0 = 0, v0 = 0, amsgrad = False, vhat(0) = 0
                if len(state) == 0:
                    state["m"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                    state["v"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                    state["vhat"] = torch.zeros_like(param, memory_format=torch.preserve_format)

                m_prev: Tensor = state["m"]
                v_prev: Tensor = state["v"]
                vhat_prev: Tensor = state["vhat"]

                # Step 4: Draw random batch from dataset is handled in the training loop.

                # Step 5: g_t <- gradient at the current parameters.
                g_t = grad

                # Step 6: m_t <- beta1 * m_{t-1} + (1 - beta1) * g_t.
                m_t = beta1 * m_prev + (1.0 - beta1) * g_t

                # Step 7: m_max <- Max(m_{t-1}, |g_t|) using Euclidean norms.
                g_abs = g_t.abs()
                m_prev_norm = torch.linalg.vector_norm(m_prev)
                g_abs_norm = torch.linalg.vector_norm(g_abs)
                m_max = torch.maximum(m_prev_norm, g_abs_norm)

                # Step 8: Lambda(t) <- Lambda_t0 - (m_{t-1} / m_max).
                # If m_max == 0, set ratio = 0 to avoid division by zero.
                zero = torch.zeros((), dtype=param.dtype, device=param.device)
                ratio = torch.where(m_max > 0.0, m_prev_norm / m_max, zero)
                lambda_t = torch.as_tensor(lambda_t0, dtype=param.dtype, device=param.device) - ratio

                # Step 9: v_t <- beta2 * v_{t-1} + (1 - beta2) * (|g_t|)^Lambda(t).
                v_t = beta2 * v_prev + (1.0 - beta2) * g_abs.pow(lambda_t)

                # Step 10: If Lambda(t) < 2 then switch to AMSGrad path.
                if bool((lambda_t < 2.0).item()):
                    # Step 11: amsgrad = True
                    group["amsgrad"] = True

                    # Step 12: vhat_t <- Max(vhat_{t-1}, |v_t|).
                    vhat_t = torch.maximum(vhat_prev, v_t.abs())
                    state["vhat"] = vhat_t

                    # Step 13: theta_t <- theta_{t-1} - eta * m_t / (vhat_t^(1/Lambda(t)) + eps).
                    denom = vhat_t.pow(1.0 / lambda_t) + eps
                else:
                    # Step 15: amsgrad = False
                    group["amsgrad"] = False

                    # Step 16: theta_t <- theta_{t-1} - eta * m_t / (v_t^(1/Lambda(t)) + eps).
                    denom = v_t.pow(1.0 / lambda_t) + eps

                param.addcdiv_(m_t, denom, value=-lr)

                state["m"] = m_t
                state["v"] = v_t

        # Step 18: return final parameters is implicit after repeated updates.
        return loss
