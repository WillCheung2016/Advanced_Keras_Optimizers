from __future__ import absolute_import
from keras import backend as K
from keras.optimizers import Optimizer

class SMORMS3(Optimizer):
    '''SMORMS3 optimizer.

    Implemented based on http://sifter.org/~simon/journal/20150420.html

    # Arguments
        lr: float >= 0. Learning rate.
        epsilon: float >= 0. Fuzz factor.
        decay: float >= 0. Learning rate decay over each update.
    '''

    def __init__(self, lr=0.001, epsilon=1e-16, decay=0.,
                 **kwargs):
        super(SMORMS3, self).__init__(**kwargs)
        self.__dict__.update(locals())
        with K.name_scope(self.__class__.__name__):
            self.lr = K.variable(lr)
            # self.rho = K.variable(rho)
            self.decay = K.variable(decay)
            self.inital_decay = decay
            self.iterations = K.variable(0.)
        self.epsilon = epsilon
        self.initial_decay = decay

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        shapes = [K.get_variable_shape(p) for p in params]
        self.updates.append(K.update_add(self.iterations, 1))

        g1s = [K.zeros(shape) for shape in shapes]
        g2s = [K.zeros(shape) for shape in shapes]
        mems = [K.ones(shape) for shape in shapes]

        lr = self.lr
        if self.inital_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))

        self.weights = [self.iterations] + g1s + g2s + mems

        for p, g, g1, g2, m in zip(params, grads, g1s, g2s, mems):
            r = 1. / (m + 1)
            new_g1 = (1. - r) * g1 + r * g
            new_g2 = (1. - r) * g2 + r * K.square(g)
            # update accumulators
            self.updates.append(K.update(g1, new_g1))
            self.updates.append(K.update(g2, new_g2))
            new_p = p - g * K.minimum(lr, K.square(new_g1) / (new_g2 + self.epsilon)) / (
            K.sqrt(new_g2) + self.epsilon)
            new_m = 1 + m * (1 - K.square(new_g1) / (new_g2 + self.epsilon))
            # update rho
            self.updates.append(K.update(m, new_m))
            # apply constraints
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon}
        base_config = super(SMORMS3, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))