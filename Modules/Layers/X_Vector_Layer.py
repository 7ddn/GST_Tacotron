import tensorflow as tf

class StatPoolingLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        means = tf.math.reduce_mean(inputs, axis = 1, keepdims=True)
        
        #stddevs = tf.math.reduce_std(inputs, axis = 1) 
        
        variances = tf.math.reduce_mean(tf.math.square(inputs - means), axis = 1)
        means = tf.squeeze(means, axis = 1)
        stddevs = tf.math.sqrt(tf.clip_by_value(variances, 1e-10, variances.dtype.max))

        return tf.concat((means, stddevs), axis=1)

class XVectorLayer(tf.keras.Model):

    def __init__(self, num_speaker = None, if_digit = False, if_output_x_vector = True, **kwargs):
        super(XVectorLayer, self).__init__(**kwargs)
    
        # num_frame_layers = 5
        # num_segment_layer = 2

        frame_layer_filters = [512, 512, 512, 512, 1500]
        frame_layer_kernels = [5, 3, 3, 1, 1]
        frame_layer_strides = [1, 2, 3, 1, 1]

        segment_layer_units = [512, 512]

        self.frame_layers = [tf.keras.layers.Conv1D(filters, kernels, strides, padding="causal", activation="relu") for filters, kernels, strides in zip(frame_layer_filters, frame_layer_kernels, frame_layer_strides)]

        self.segment_layers = [tf.keras.layers.Dense(units, activation="relu")
         for units in segment_layer_units]

        self.stat_pooling = StatPoolingLayer()
        
        self.num_speaker = num_speaker

        if self.num_speaker is not None:
            self.output_layer = tf.keras.layers.Dense(num_speaker, activation = 'softmax')
    
        self.if_digit = if_digit

        self.if_output_x_vector = if_output_x_vector
    
    def load_cp(self, checkpoint_path):
        latest = tf.train.latest_checkpoint(checkpoint_path)
        if latest is not None:
            print(f'Found latest checkpoint for X Vector at {latest}')
        else:
            raise Exception('X Vector checkpoint not found')
        self.load_weights(latest)

    def call(self, inputs):
        x = inputs

        for layer in self.frame_layers: 
            x = layer(x)

        x = self.stat_pooling(x)

        x_vector = self.segment_layers[0](x)
        
        if self.num_speaker is None:
            return x_vector

        for layer in self.segment_layers:
            x = layer(x)
        
        output = self.output_layer(x)

        if self.if_digit:
            output = tf.math.argmax(output, axis=-1)

        if self.if_output_x_vector:
            return output, x_vector
        else:
            return output

    

