import math
import pickle
import tensorflow as tf
import numpy as np


class PPG_RNN(tf.keras.Model):
    def __init__(self, tokenizer=None, token_count = None, factor=1, mel_dims=80):
        super().__init__()
        self.tokenizer = tokenizer

        if token_count is None and tokenizer is not None:
            self.token_count = tokenizer.vocabulary_size()

        self.factor = factor
        self.mel_dims = mel_dims
        # self.output_final = output_final

        dense_units = 512
        # tf.keras.layers.Flatten(),
        self.masking = tf.keras.layers.Masking(mask_value=0.0)
        self.bidir = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=128, return_sequences=True)
        )
        self.dense_0 = tf.keras.layers.Dense(units=dense_units, activation="relu")
        # tf.keras.layers.Reshape([-1, self.factor * dense_units])
        self.avg_pool = tf.keras.layers.AveragePooling1D(
            pool_size=self.factor, strides=self.factor, padding="same"
        )
        self.lstm = tf.keras.layers.LSTM(units=128, return_sequences=True)
        if self.token_count is not None:
            self.output_dense = tf.keras.layers.Dense(
                units=self.token_count, activation="softmax"
            )
        # self.softmax = tf.keras.layers.Softmax(axis = -1)

    def call(self, mel):
        # spec shape [batch_size, num_frames + 1, fft_size]
        new_Tensor = self.masking(mel)
        new_Tensor = self.bidir(mel)
        new_Tensor = self.dense_0(new_Tensor)
        if self.factor != 1:
            new_Tensor = self.avg_pool(new_Tensor)
        new_Tensor = self.lstm(new_Tensor)
        if self.token_count is None:
            return new_Tensor

        new_Tensor = self.output_dense(new_Tensor)
        # new_Tensor = self.softmax(new_Tensor)

        return new_Tensor


class BDLayer(tf.keras.layers.Layer):
    def __init__(self, count = None, file_path = None):
        # count is a dict with key being phoneme and value
        # being counts

        if file_path is not None and count is None:
            with open(file_path, 'rb') as f:
                count = pickle.load(f)

        self.num_class = len(count)
        self.prob = [math.log(count[p] / sum(count.values())) for p in count]

        self.a = tf.convert_to_tensor(np.tile(self.prob, (self.num_class, 1)))
        self.b = tf.transpose(self.a)

        self.a = 0.5 * tf.math.log(self.a)
        self.b = 0.5 * tf.math.log(self.b)

    def call(self, ppg):
        # ppg [batch, time, dict_dim]

        time_length = tf.shape(ppg)[1]
        sim_mat = tf.matmul(tf.transpose(ppg, perm=[0, 2, 1]), ppg)
        sim_mat = -tf.math.log(tf.math.sqrt(sim_mat / time_length))

        sim_mat += self.a
        sim_mat += self.b
        # sim_mat [batch, dict_dim, dict_dim]

        return sim_mat
