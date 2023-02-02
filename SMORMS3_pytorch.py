# Conversion from Keras to Pytorch done by ChatGPT. Use with caution.
import torch
from torch.optim import Optimizer


class SMORMS3(Optimizer):
    def __init__(self, params, lr=1e-3, epsilon=1e-16, weight_decay=0, **kwargs):
        defaults = dict(lr=lr, epsilon=epsilon, weight_decay=weight_decay)
        super(SMORMS3, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            lr = group['lr']
            epsilon = group['epsilon']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                if len(state) == 0:
                    state['g1'] = torch.zeros_like(p.data)
                    state['g2'] = torch.zeros_like(p.data)
                    state['m'] = torch.ones_like(p.data)
                    state['t'] = 0

                g = p.grad.data
                g1 = state['g1']
                g2 = state['g2']
                m = state['m']
                t = state['t']

                r = 1. / (m + 1)
                new_g1 = (1 - r) * g1 + r * g
                new_g2 = (1 - r) * g2 + r * g ** 2
                new_m = 1 + m * (1 - g1 ** 2 / (g2 + epsilon))
                t += 1

                p.data.add_(-lr * g * torch.min(1, g1 ** 2 / (g2 + epsilon)) / (torch.sqrt(g2) + epsilon))
                p.data.add_(-weight_decay * p.data)  # weight decay

                state['g1'].copy_(new_g1)
                state['g2'].copy_(new_g2)
                state['m'].copy_(new_m)
                state['t'] = t

        return loss

