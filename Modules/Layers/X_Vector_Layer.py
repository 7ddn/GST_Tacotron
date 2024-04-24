import tensorflow as tf

class StatPoolingLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        means = tf.math.reduce_mean(inputs, axis = 1)
        stddevs = tf.math.reduce_std(inputs, axis = 1) 
        return tf.concat((means, stddevs), axis=1)

class XVectorLayer(tf.keras.layers.Layer):

    # TODO: Read parameters from Hparamters file
    def __init__(self, num_speaker, if_digit = False,**kwargs):
        super(XVectorLayer, self).__init__(**kwargs)
    
        # num_frame_layers = 5
        # num_segment_layer = 2

        frame_layer_filters = [512, 512, 512, 512, 1500]
        frame_layer_kernels = [5, 3, 3, 1, 1]
        frame_layer_strides = [1, 2, 3, 1, 1]

        segment_layer_units = [512, 512]

        self.frame_layers = [tf.keras.layers.Conv1D(filters, kernels, strides) for filters, kernels, strides in zip(frame_layer_filters, frame_layer_kernels, frame_layer_strides)]

        self.segment_layers = [tf.keras.layers.Dense(units) for units in segment_layer_units]

        self.stat_pooling = StatPoolingLayer()

        self.output_layer = tf.keras.layers.Dense(num_speaker, activation = 'log_softmax')
    
        self.if_digit = if_digit
    def call(self, inputs):
        x = inputs

        for layer in self.frame_layers: 
            x = layer(x)

        x = self.stat_pooling(x)

        x_vector = self.segment_layers[0](x)

        for layer in self.segment_layers:
            x = layer(x)

        output = self.output_layer(x)

        if self.if_digit:
            output = tf.math.argmax(output, axis=-1)

        return output, x_vector



