from __future__ import absolute_import

from keras import backend as K
from keras.optimizers import Optimizer

if K.backend() == 'tensorflow':
    import tensorflow as tf

class COCOB(Optimizer):
    """COCOB-Backprop optimizer.

    It is recommended to leave the parameters of this optimizer
    at their default values
    (except the learning rate, which can be freely tuned).

    This optimizer, unlike other stochastic gradient based optimizers, optimize the function by
    finding individual learning rates in a coin-betting way.

    # Arguments
        alphs: float >= 0. Multiples of the largest absolute magtitude of gradients.
        epsilon: float >= 0. Fuzz factor.

    # References
        - [Training Deep Networks without Learning Rates Through Coin Betting](http://https://arxiv.org/pdf/1705.07795.pdf)
    """

    def __init__(self, alpha=100, epsilon=1e-8, **kwargs):
        super(COCOB, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.alpha = K.variable(alpha, name='alpha')
            self.iterations = K.variable(0., name='iterations')
        self.epsilon = epsilon

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        L = [K.zeros(K.get_variable_shape(p), dtype=K.dtype(p)) for p in params]
        M = [K.zeros(K.get_variable_shape(p), dtype=K.dtype(p)) for p in params]
        Reward = [K.zeros(K.get_variable_shape(p), dtype=K.dtype(p)) for p in params]
        grad_sum = [K.zeros(K.get_variable_shape(p), dtype=K.dtype(p)) for p in params]

        if K.eval(self.iterations) == 0:
            old_params = [K.constant(K.eval(p)) for p in params]
            # [K.eval(p) for p in params]

        self.weights = [self.iterations] + L + M + Reward + grad_sum

        for old_p, p, g, gs, l, m, r in zip(old_params, params, grads, grad_sum, L, M, Reward):
            # update accumulator
            # old_p = K.variable(old_p)

            new_l = K.maximum(l, K.abs(g))
            self.updates.append(K.update(l, new_l))

            new_m = m + K.abs(g)
            self.updates.append(K.update(m, new_m))

            new_r = K.maximum(r - (p - old_p)*g, 0)
            self.updates.append(K.update(r, new_r))

            new_gs = gs + g
            self.updates.append(K.update(gs, new_gs))

            new_p = old_p - (new_gs/(self.epsilon + new_l*K.maximum(new_m+new_l, self.alpha*new_l)))*(new_l + new_r)

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'alpha': float(K.get_value(self.alpha)),
                  'epsilon': self.epsilon}
        base_config = super(COCOB, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))