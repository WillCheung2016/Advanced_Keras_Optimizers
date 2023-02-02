# Conversion from Keras to Pytorch done by ChatGPT. Use with caution.
import torch

class SMORMS3:
    def __init__(self, params, lr=0.001, epsilon=1e-16, decay=0):
        self.params = list(params)
        self.lr = lr
        self.epsilon = epsilon
        self.decay = decay
        self.iterations = 0

        self.g1s = [torch.zeros_like(p) for p in self.params]
        self.g2s = [torch.zeros_like(p) for p in self.params]
        self.mems = [torch.ones_like(p) for p in self.params]

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        self.iterations += 1

        lr = self.lr * (1. / (1. + self.decay * self.iterations))

        for p, g1, g2, m in zip(self.params, self.g1s, self.g2s, self.mems):
            g = p.grad.data
            r = 1. / (m + 1)
            new_g1 = (1. - r) * g1 + r * g
            new_g2 = (1. - r) * g2 + r * g.pow(2)
            p.data -= g * torch.min(lr, new_g1.pow(2) / (new_g2 + self.epsilon)) / (
                    torch.sqrt(new_g2) + self.epsilon)

            g1.data = new_g1
            g2.data = new_g2
            m.data = 1 + m * (1 - new_g1.pow(2) / (new_g2 + self.epsilon))

        return loss
