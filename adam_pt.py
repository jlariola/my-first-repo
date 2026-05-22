import torch
from torch.optim.optimizer import Optimizer

class Adam(Optimizer):
    """
    Implements Adam algorithm.
    
    This implementation strictly adheres to the pseudocode provided in the 
    official PyTorch documentation and the original paper (Kingma & Ba, 2014).
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, amsgrad=False, maximize=False):
        
        # Handle edge cases and invalid configurations
        if params is None:
            raise ValueError("Invalid params: None")
        if betas is None or not isinstance(betas, (tuple, list)) or len(betas) != 2:
            raise ValueError("Invalid betas: expected a tuple of two floats")
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        amsgrad=amsgrad, maximize=maximize)
        super(Adam, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # Extract hyperparameters for the group
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            amsgrad = group['amsgrad']
            maximize = group['maximize']

            for p in group['params']:
                # Skip parameters with no gradients
                if p.grad is None:
                    continue
                
                if p.grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients.")

                # g_t = grad(theta_{t-1})
                grad = p.grad
                
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # m_0 = 0 (Exponential moving average of gradient values)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # v_0 = 0 (Exponential moving average of squared gradient values)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # v_max_0 = 0 (Maintains max of all exp. moving avg. of sq. grad. values)
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                # t = t + 1
                state['step'] += 1
                t = state['step']

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']

                g_t = grad

                # if maximize: g_t = -g_t
                if maximize:
                    g_t = -g_t

                # if weight_decay != 0: g_t = g_t + lambda * theta_{t-1}
                if weight_decay != 0:
                    g_t = g_t.add(p, alpha=weight_decay)

                # m_t = beta_1 * m_{t-1} + (1 - beta_1) * g_t
                exp_avg.mul_(beta1).add_(g_t, alpha=1.0 - beta1)

                # v_t = beta_2 * v_{t-1} + (1 - beta_2) * g_t^2
                exp_avg_sq.mul_(beta2).addcmul_(g_t, g_t, value=1.0 - beta2)

                # if amsgrad: v_hat_t = max(v_hat_{t-1}, v_t)
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                    # v_t^max = max(v_t^max, v_t)
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # v_hat_t = v_t^max
                    v_hat_t = max_exp_avg_sq
                # else: v_hat_t = v_t
                else:
                    # v_hat_t = v_t
                    v_hat_t = exp_avg_sq

                # bias_correction1 = 1 - beta_1^t
                bias_correction1 = 1.0 - beta1 ** t
                
                # bias_correction2 = 1 - beta_2^t
                bias_correction2 = 1.0 - beta2 ** t

                # m_hat_t = m_t / (1 - beta_1^t)
                m_hat_t = exp_avg / bias_correction1
                # v_hat_t = v_hat_t / (1 - beta_2^t)
                v_hat_t = v_hat_t / bias_correction2

                # theta_t = theta_{t-1} - lr * m_hat_t / (sqrt(v_hat_t) + eps)
                denom = v_hat_t.sqrt().add_(eps)
                p.addcdiv_(m_hat_t, denom, value=-lr)

        return loss