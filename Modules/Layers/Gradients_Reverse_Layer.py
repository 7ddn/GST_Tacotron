# Forked from https://gist.github.com/oO0oO0oO0o0o00/74dbcb352164348e5268203fdf95a04b

import tensorflow as tf

class GradientReversalLayer(tf.keras.layers.Layer):
    def __init__(self, lamb=1, **kwargs):
        super(self).__init(**kwargs)
        self.lamb = lamb
        
        @tf.custom_gradient
        def reverse_gradient(x, lamb):
            return tf.identity(x), lambda dy: (-lamb*dy, None)

        def call(self, x):
            return self.reverse_gradient(x, self.lamb)

        



