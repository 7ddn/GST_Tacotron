import tensorflow as tf
import json
from .Attention.Layers import MultiHeadAttention
from Modules.ppg_train import PPG_CNN
# import PPG_CNN
import os
import pickle
import math

with open('Hyper_Parameters.json', 'r') as f:
    hp_Dict = json.load(f)

with open(hp_Dict['Token_JSON_Path'], 'r') as f:
    token_Index_Dict = json.load(f)

class Reference_Encoder(tf.keras.Model):
    def __init__(self):        
        super(Reference_Encoder, self).__init__()
        self.layer_Dict = {}

        self.layer_Dict['RNN'] = tf.keras.layers.GRU(
            units= hp_Dict['GST']['Reference_Encoder']['RNN']['Size'],
            return_sequences= True
            )

        self.layer_Dict['Compress_Length'] = tf.keras.layers.Lambda(
            lambda x: tf.cast(tf.math.ceil(x / tf.reduce_prod(hp_Dict['GST']['Reference_Encoder']['Conv']['Strides'])), tf.int32)
            )

        

        self.layer_Dict['Dense'] = tf.keras.layers.Dense(
            units= hp_Dict['GST']['Reference_Encoder']['Dense']['Size'],
            activation= 'tanh'
            )
    
        tv_dict_path = "tv_layer_new.pkl"

        # TODO: move path to hp file
        from_disk = pickle.load(open(tv_dict_path, "rb"))
        token_layer = tf.keras.layers.TextVectorization.from_config(from_disk['config'])
        token_layer.set_weights(from_disk['weights'])

        ppg_layer = PPG_CNN(token_layer, factor = 1)
        ppg_layer.trainable = False
        cp_dir = './ppg/checkpoints'
        latest = tf.train.latest_checkpoint(cp_dir)
        ppg_layer.load_weights(latest).expect_partial()

        self.layer_Dict['PPG'] = ppg_layer
        print(f'Successfully loaded ppg checkpoint from {latest}') 



        self.ppg_dim = len(token_layer.get_vocabulary())
        
        # Use a resnet style skip connection to avoid ppg sparse issues
        for num in range(hp_Dict['GST']['Reference_Encoder']['SkipConnection']['Numbers']):
            depth = len(hp_Dict['GST']['Reference_Encoder']['SkipConnection']['Sizes'])
            for index, units in enumerate(hp_Dict['GST']['Reference_Encoder']['SkipConnection']['Sizes']):
                self.layer_Dict[f'Skip_Dense_{num*(depth+1)+index}'] = tf.keras.layers.Dense(
                    units = units,
                    activation = 'relu',
                    use_bias = False
                )
            self.layer_Dict[f'Skip_Dense_{num*(depth+1)+index+1}'] = tf.keras.layers.Dense(
                    units = self.ppg_dim,
                    activation = 'relu',
                    use_bias = False
                )
                

        # Goal: Use PPG, instead of raw mel, to feed reference encoder

        # RNN Decoder, processing ppg to some embedding
        self.layer_Dict['Encoder_RNN'] = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                units = hp_Dict['GST']['Reference_Encoder']['PPG_RNN']['Size']))
        
    def call(self, inputs):
        '''

        inputs: [mels, mel_lengths]
        mels: [Batch, Time, Mel_Dim]
        mel_lengths: [Batch]
        '''
        mels, mel_lengths = inputs
 
        ppgs = self.layer_Dict['PPG'](mels) #[Batch, Time, PPG_Dim]

        # Apply mask to ppg to make sure padded zeros are also zeros in ppg
        
        mask = tf.cast(mels == 0.0, ppgs.dtype)[:, :, 0:self.ppg_dim]
        ppgs = tf.multiply(ppgs, mask)
        
        for num in range(hp_Dict['GST']['Reference_Encoder']['SkipConnection']['Numbers']): 
            new_Tensor = ppgs
            for index in range(len(hp_Dict['GST']['Reference_Encoder']['SkipConnection']['Sizes'])+1):
                new_Tensor = self.layer_Dict[f'Skip_Dense_{index}'](new_Tensor)
            new_Tensor += ppgs
            ppgs = new_Tensor

        # [Batch, Time, PPG_Dim]

        new_Tensor = self.layer_Dict['Encoder_RNN'](new_Tensor) #[Batch, RNN_Dim]

        return self.layer_Dict['Dense'](new_Tensor)

class Style_Token_Layer(tf.keras.layers.Layer): #Attention which is in layer must be able to access directly.
    def __init__(self):
        super(Style_Token_Layer, self).__init__()
        
    def build(self, input_shape):        
        self.layer_Dict = {}
        self.layer_Dict['Reference_Encoder'] = Reference_Encoder()
        self.layer_Dict['Attention'] = MultiHeadAttention(
            num_heads= hp_Dict['GST']['Style_Token']['Attention']['Head'],
            size= hp_Dict['GST']['Style_Token']['Attention']['Size']
            )

        self.gst_tokens = self.add_weight(
            name= 'gst_tokens',
            shape= [hp_Dict['GST']['Style_Token']['Size'], hp_Dict['GST']['Style_Token']['Embedding']['Size']],
            initializer= tf.keras.initializers.TruncatedNormal(stddev= 0.5),
            trainable= True,
            )

    def call(self, inputs, if_mean = False):
        '''
        inputs: [mels, mel_lengths]
        mels: [Batch, Time, Mel_Dim]
        mel_lengths: [Batch]
        '''
        mels_for_gst, mel_lengths = inputs        
        new_Tensor = self.layer_Dict['Reference_Encoder']([mels_for_gst[:, 1:], mel_lengths])  #Initial frame deletion

        tiled_GST_Tokens = tf.tile(
            tf.expand_dims(tf.tanh(self.gst_tokens), axis=0),
            [tf.shape(new_Tensor)[0], 1, 1]
            )   #[Token_Dim, Emedding_Dim] -> [Batch, Token_Dim, Emedding_Dim]
        new_Tensor = tf.expand_dims(new_Tensor, axis= 1)    #[Batch, R_dim] -> [Batch, 1, R_dim]
        new_Tensor, _ = self.layer_Dict['Attention'](
            inputs= [new_Tensor, tiled_GST_Tokens]  #[query, value]
            )   #[Batch, 1, Att_dim]
        
        return tf.squeeze(new_Tensor, axis= 1)

class L2UnitRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, l2 = 0.):
        self.l2 = tf.keras.regularizers.L2(l2 = l2)
    def __call__(self, x):
        return self.l2(x - tf.eye(x.shape[-1]))
        

class GST_Phoneme_Encoder(tf.keras.layers.Layer):

    def __init__(self):
        super(GST_Phoneme_Encoder, self).__init__()

        self.layer_Dict = {}
        
        for i in range(hp_Dict['GST']['Style_Token']['Attention']['Size']):
            self.layer_Dict[f'Dense_{i}'] = tf.keras.layers.Dense(
                # units = hp_Dict['GST']['Phoneme_Layer']['Depth'],
                units = hp_Dict['Tacotron2']['Encoder']['Embedding']['Size'],
                kernel_initializer = tf.keras.initializers.Identity(),
                kernel_regularizer = L2UnitRegularizer(1e-4),
                use_bias = False,
            )

        # self.layer_Dict['Flatten'] = tf.keras.layers.Flatten()

    def call(self, inputs):
        '''
        inputs: [encoder, gsts]
        '''
        encoders, gsts = inputs
        # encoders # [Batch, Time_Step, Dim]       
        # GST [Batch, GST_Dim]

        gsts = tf.nn.softmax(gsts[:, :, tf.newaxis, tf.newaxis], axis = 1) # [Batch, GST_Dim, 1, 1]

        embedded = [self.layer_Dict[f'Dense_{i}'](encoders) for i in range(hp_Dict['GST']['Style_Token']['Attention']['Size'])] # GST_Dim * [Batch, Time_Step, new_Dim]
        embedded = tf.stack(embedded, axis = 1) # [Batch, GST_Dim, Time_Step, new_Dim]

        new_Tensor = gsts * embedded # [Batch, GST_Dim, Time_Step, Depth]
        new_Tensor = tf.reduce_sum(new_Tensor, axis = 1)

        # output [Batch, Time_Step, Output_Dim]
        
        return new_Tensor

        ''' 
        return tf.concat([
            tf.tile(tf.expand_dims(gsts, axis= 1), [1, tf.shape(encoders)[1], 1]),
            encoders
            ], axis= -1)
        '''
