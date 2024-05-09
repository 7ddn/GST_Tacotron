import tensorflow as tf

class XVectorConcatLayer(tf.keras.layers.Layer):
    def __init__(self, new_Dimension, **kwargs):
        super(XVectorConcatLayer, self).__init__(**kwargs)

        self.resize_layer = tf.keras.layers.Dense(units = new_Dimension)

    def call(self, inputs):
        '''
        input: [encoders, xvectors]
        '''

        encoders, xvectors = inputs
        resized_xvectors = self.resize_layer(xvectors)

        return tf.concat([
            tf.tile(tf.expand_dims(resized_xvectors, axis=1), [1, tf.shape(encoders)[1], 1]),
            encoders
            ], axis = -1)
