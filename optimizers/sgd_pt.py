import torch
from torch.optim.optimizer import Optimizer
from typing import Iterable, Optional, Callable

class SGD(Optimizer):
    """
    Implements Stochastic Gradient Descent (optionally with momentum, Nesterov, and weight decay).
    
    This implementation strictly translates the official PyTorch 2.12.0 pseudocode logic
    into an efficient, executable Optimizer format.
    """
    def __init__(
        self, 
        params: Iterable[torch.Tensor], 
        lr: float, 
        momentum: float = 0.0, 
        dampening: float = 0.0,
        weight_decay: float = 0.0, 
        nesterov: bool = False, 
        maximize: bool = False
    ):
        # Handle edge cases: Validate hyperparameter inputs
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if dampening < 0.0:
            raise ValueError(f"Invalid dampening value: {dampening}")
        if nesterov and (momentum <= 0.0 or dampening != 0.0):
            raise ValueError("Nesterov momentum requires a positive momentum and zero dampening")

        defaults = dict(
            lr=lr, momentum=momentum, dampening=dampening,
            weight_decay=weight_decay, nesterov=nesterov, maximize=maximize
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # for t=1 to ... do (Implied loop driven by successive calls to optimizer.step())
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            maximize = group['maximize']
            lr = group['lr']

            for p in group['params']:
                # Handle edge case: Skip parameter if there is no gradient available
                if p.grad is None:
                    continue
                    
                # if maximize:
                #     g_t = -gradient
                # else:
                #     g_t = gradient
                g_t = -p.grad if maximize else p.grad

                # if g_t is sparse: only the plain update is supported
                if g_t.is_sparse:
                    if weight_decay != 0.0 or momentum != 0.0 or nesterov:
                        raise RuntimeError(
                            "Sparse gradients do not support weight_decay, momentum, or nesterov"
                        )
                    # theta_t = theta_{t-1} - lr * g_t
                    p.add_(g_t, alpha=-lr)
                    continue

                # if weight_decay != 0:
                #     g_t = g_t + weight_decay * theta_{t-1}
                if weight_decay != 0.0:
                    g_t = g_t.add(p, alpha=weight_decay)

                # if momentum != 0:
                if momentum != 0.0:
                    param_state = self.state[p]
                    
                    # if t > 1:
                    #     b_t = momentum * b_{t-1} + (1 - dampening) * g_t
                    # else:
                    #     b_t = g_t
                    if 'momentum_buffer' not in param_state:
                        # Initialization at t=1: b_t = g_t
                        b_t = param_state['momentum_buffer'] = torch.clone(g_t).detach()
                    else:
                        # Subsequent steps t > 1: momentum calculation
                        b_t = param_state['momentum_buffer']
                        # Modifies buffer in-place for maximum efficiency
                        b_t.mul_(momentum).add_(g_t, alpha=1.0 - dampening)

                    # if nesterov:
                    #     g_t = g_t + momentum * b_t
                    # else:
                    #     g_t = b_t
                    if nesterov:
                        g_t = g_t.add(b_t, alpha=momentum)
                    else:
                        g_t = b_t

                # theta_t = theta_{t-1} - lr * g_t
                p.add_(g_t, alpha=-lr)

        return loss