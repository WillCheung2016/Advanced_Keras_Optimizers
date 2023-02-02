# Implemented by ChatGPT. Use with caution.
import torch
from torch.optim import Optimizer

class SMORMS3(Optimizer):
    def __init__(self, params, lr=1e-3, epsilon=1e-16, decay=0, weight_decay=0):
        defaults = dict(lr=lr, epsilon=epsilon, decay=decay, weight_decay=weight_decay)
        super(SMORMS3, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            lr = group['lr']
            epsilon = group['epsilon']
            decay = group['decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                if len(state) == 0:
                    state['g1'] = torch.zeros_like(p.data)
                    state['g2'] = torch.zeros_like(p.data)
                    state['m'] = torch.ones_like(p.data)
                else:
                    g1, g2, m = state['g1'], state['g2'], state['m']

                r = 1. / (m + 1)
                new_g1 = (1 - r) * g1 + r * grad
                new_g2 = (1 - r) * g2 + r * grad * grad
                new_m = 1 + m * (1 - new_g1 * new_g1 / (new_g2 + epsilon))

                g1.copy_(new_g1)
                g2.copy_(new_g2)
                m.copy_(new_m)

                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)

                lr_ = lr
                if decay > 0:
                    lr_ *= (1. / (1. + decay * m))

                p.data.add_(-lr_ * new_g1 / (torch.sqrt(new_g2) + epsilon))

        return loss

