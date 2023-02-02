# Implemented by ChatGPT. Use with caution.
import torch
import math
from torch.optim.optimizer import Optimizer

class SMORMS3(Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0, epsilon=1e-16, betas=(0.9, 0.999)):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= epsilon:
            raise ValueError("Invalid epsilon value: {}".format(epsilon))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        defaults = dict(lr=lr, weight_decay=weight_decay, epsilon=epsilon, betas=betas)
        super(SMORMS3, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('SMORMS3 does not support sparse gradients')

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['g1'] = torch.zeros_like(p.data)
                    state['g2'] = torch.zeros_like(p.data)
                    state['m'] = torch.ones_like(p.data)

                g1 = state['g1']
                g2 = state['g2']
                m = state['m']
                r = 1. / (m + 1)
                new_g1 = (1. - r) * g1 + r * grad
                new_g2 = (1. - r) * g2 + r * grad * grad
                new_m = 1 + m * (1 - new_g1 * new_g1 / (new_g2 + group['epsilon']))
                lr = group['lr']
                weight_decay = group['weight_decay']

                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)

                p.data.addcdiv_(-lr, new_g1, (new_g2 + group['epsilon']).sqrt())
                state['step'] += 1
                state['g1'] = new_g1
                state['g2'] = new_g2
                state['m'] = new_m

        return loss
