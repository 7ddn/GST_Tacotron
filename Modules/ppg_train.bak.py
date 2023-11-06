''' Use a simple CNN for test, should use something like transformer for further research. '''

import tensorflow as tf
import os
from tqdm import tqdm
import pickle

import argparse

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import time

parser = argparse.ArgumentParser(description='mode')
parser.add_argument('--mode', type=str, help='mode of script, train or eval')
parser.add_argument('--factor')
parser.add_argument('--dir')
parser.add_argument('--sr')
parser.add_argument('--frame_step')
parser.add_argument('--frame_length')
# parser.add_argument(['-c', '--continue'], action = 'store_false')

args = parser.parse_args()
mode = args.mode
dir = args.dir
if dir is None:
    dir = ''
if args.factor is None:
    factor = 4
else:
    factor = int(args.factor)
if args.sr is None:
    sr = 16000
else:
    sr = int(args.sr)
if args.frame_step is None:
    frame_step = 80
else:
    frame_step = int(args.frame_step)
if args.frame_length is None:
    frame_length = 160
else:
    frame_length = int(args.frame_length)


'''
# factor = 4
base_frame_length = 320
base_frame_step = 160
frame_length_factor = 2

frame_length = base_frame_length / frame_length_factor
frame_step = base_frame_step / factor 
'''

def remove_letter(text):
    return tf.strings.regex_replace(text, f'[0-9]', '')

def init():

    database_dir = os.path.expanduser(dir)

    ds = tf.data.Dataset.load(database_dir)


    tv_dict_path = "tv_layer_new.pkl"
    # TODO: move path to hyperparameter file

    if os.path.isfile(tv_dict_path):
        from_disk = pickle.load(open(tv_dict_path, "rb"))
        token_layer = tf.keras.layers.TextVectorization.from_config(from_disk['config'])
        token_layer.set_weights(from_disk['weights'])
    else:
        token_layer = tf.keras.layers.TextVectorization(standardize = remove_letter)
        token_layer.adapt(ds.map(lambda wav, phone, speaker, filename: phone))
        pickle.dump({'config':token_layer.get_config(), 'weights': token_layer.get_weights()}, open(tv_dict_path, "wb"))

    print(f'Loaded Vocabulary with length {len(token_layer.get_vocabulary())}')
    
    return token_layer, ds


def get_mel(wav, sr = 16000, frame_length = 160, frame_step = 80, factor = 1):
    sp = tf.signal.stft(wav, frame_length = int(frame_length), frame_step = int(frame_step), pad_end = True)
    sp = tf.abs(sp)
    # print(sp.shape)
    num_sp_bins = sp.shape[-1]
    leh, ueh, num_bins = 80.0 / 16000 * sr, 7600.0 / 16000 * sr, 80
    mat = tf.signal.linear_to_mel_weight_matrix(
    num_bins, num_sp_bins, sr, leh, ueh)
    mel = tf.tensordot(sp, mat, 1)
    mel.set_shape(sp.shape[:-1].concatenate(mat.shape[-1:]))
    log_mel = tf.math.log(mel + 1e-6)
    # print(log_mel.shape)
    log_mel = tf.reshape(log_mel, [-1, factor, num_bins])
    # print(log_mel.shape)
    return log_mel#[:-1, :]

class PPG_CNN(tf.keras.Model):
    def __init__(self, tokenizer, factor = 4, mel_dim = 80):
        super().__init__()
        self.factor = factor
        self.mel_dim = mel_dim
        self.tokenizer = tokenizer
        conv_fil = [32, 32, 64, 64, 128, 128]
        conv_kernel = [3, 3, 3, 3, 3, 3]
        conv_stride = [2, 2, 2, 2, 2, 2]

        self.conv_layers = []
        for filters, kernel_size, strides in zip(conv_fil, conv_kernel, conv_stride):
            self.conv_layers.append(tf.keras.layers.Conv2D(
                filters = filters,
                kernel_size = kernel_size,
                strides = strides,
                padding = 'same',
                use_bias = False
            ))
            self.conv_layers.append(tf.keras.layers.BatchNormalization())
            self.conv_layers.append(tf.keras.layers.ReLU())
        self.rnn_layer = tf.keras.layers.GRU(units = 128, return_sequences = True)
        
        self.Output = tf.keras.models.Sequential(
            [tf.keras.layers.Dense(units = self.tokenizer.vocabulary_size()),
            tf.keras.layers.Softmax(axis = -1)]
        )
    def call(self, mel):
        # mel shape [Time_Step, factor, mel_dim]
        
        new_tensor = tf.reshape(mel, [1, -1, self.factor ,self.mel_dim])
        # input shape [Batch_Size(1), Time_Step, Factor*Mel_Dim, 1]    
           
        for layer in self.conv_layers:
            new_tensor = layer(new_tensor)
        
        batch_size, time_step = tf.shape(new_tensor)[0], tf.shape(new_tensor)[1]
        height, width = new_tensor.get_shape().as_list()[2:]
        
        conv_output =  tf.reshape(new_tensor, [batch_size, time_step, height*width])

        rnn_output = self.rnn_layer(conv_output)

        ppg = self.Output(rnn_output)
        return ppg

def init_ds():
    
    token_ds = ds.map(lambda wav, phone, speaker, filename: (get_mel(wav, frame_length=frame_length, frame_step=frame_step, sr = sr, factor=factor), token_layer(phone)[:, 0]))#.map(lambda mel, token: (mel[:min(mel.shape[0], token.shape[0])], token[min(mel.shape[0], token.shape[0])]))


    def split_dataset(ds, ds_size=None, train_split = 0.8, val_split = 0.1, test_split = 0.1, shuffle = True, shuffle_size = 10000):
        assert (train_split + test_split + val_split == 1)

        # ds =

        if ds_size is None:
            ds_size = len(ds)

        if shuffle:
            ds = ds.shuffle(buffer_size = shuffle_size, seed = 42)

        train_size = int(train_split * ds_size)
        val_size = int(val_split * ds_size)
        
        train_ds = ds.take(train_size)
        val_ds = ds.skip(train_size).take(val_size)
        test_ds = ds.skip(train_size).skip(val_size)

        return train_ds, val_ds, test_ds

    train_ds, val_ds, test_ds = split_dataset(token_ds)

    train_ds = train_ds#.batch(32)
    val_ds = val_ds#.batch(32)
    test_ds = test_ds#.batch(32)

    print('Dataset Generated:\n Training Dataset Size: {train_size}\n Validation Dataset Size: {val_size}\n Test Dataset Size: {test_size}'.format(
        train_size = len(train_ds), val_size = len(val_ds), test_size = len(test_ds)))

    return train_ds, val_ds, test_ds

def train_model(masked_token = None):
    '''
    if masked_token is None:
       masked_token = None
    '''

    @tf.function
    def if_masked(token):
        # print(token)
        # print(f'token.dtype: {token.dtype}, masked_token.dtype: {masked_token.dtype}')
        # token = tf.cast(token, masked_token.dtype) 
        if token == masked_token: # or token[0] == masked_token:
            return tf.constant(0, dtype = tf.int64)
        else: 
            return tf.constant(1, dtype = tf.int64)
    

    def masked_loss(y_truth, y_pred):
        # y_pred ppg shape [batch_size, num_frames, num_phoneme]
        # y_truth shape [batch_size, num_frames]
        loss = 0.    

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(reduction = 'none')
         
        loss = loss_fn(y_truth, y_pred)
        mask = tf.map_fn(fn = if_masked, elems = y_truth)
        mask = tf.cast(mask, loss.dtype) 
        
        loss = loss*mask
        loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
        # for y_t, y_p in zip(y_truth, y_pred):
            # loss = loss + loss_fn(y_truth, y_pred) 
        # print(len(y_truth))
        # print(len(y_pred))
        # min_len = min(y_truth.shape[0], y_pred.shape[0])    

        return loss

    def masked_acc(y_truth, y_pred):
        # y_pred ppg shape [batch_size, num_frames, num_phoneme]
        # y_truth shape [batch_size, num_frames]    

        y_truth = tf.cast(y_truth, tf.int64)
        mask = tf.map_fn(fn = if_masked, elems = y_truth)
        # mask = y_truth not in masked_token
        p_pred = tf.cast(tf.math.argmax(y_pred, axis = -1), tf.int64)
        match = tf.cast(p_pred == tf.squeeze(y_truth), tf.int64)
        # print(f'{tf.shape(y_truth)}, {tf.shape(match)}, {tf.shape(mask)}')
        masked_match = match * mask
        acc = tf.reduce_sum(masked_match)/tf.reduce_sum(mask)
        
        return acc

    # model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = 'sparse_categorical_accuracy')
    model.compile(optimizer = 'adam', loss = masked_loss, metrics = masked_acc) 


    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath = './ppg/checkpoints/ckp.{epoch:02d}',
        save_weights_only = True,
        save_best_only = True,
        verbose = 1)

    history = model.fit(
        train_ds,
        epochs = 100,
        validation_data = val_ds,
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=5), checkpoint_callback])

    print('Evaluating trained model')
    model.evaluate(test_ds) 

def eval_model(token_layer):
    model = PPG_CNN(tokenizer=token_layer)
    cp_dir = './checkpoints'
    latest = tf.train.latest_checkpoint(cp_dir)
    if latest is not None:
        print(f'Found latest checkpoint at {latest}')
    else:
        raise Exception('Checkpoint not found')
    model.load_weights(latest).expect_partial()
    
    print('model is loaded')    

    dict = token_layer.get_vocabulary()
    print('read dictionary')

    for sample in test_ds:
        # print(sample)
        mel, token = sample
        break
    print('successfully taken 1 sample from test dataset')

    pred = model(mel)
    ph_pred = [dict[i.numpy()] for i in tf.argmax(pred, axis=-1)]

    ph_truth = [dict[int(i)] for i in token.numpy()]

    print(f'prediction: {ph_pred}')
    print(f'ground truth: {ph_truth}')
    #print(token)
    fig, ax = plt.subplots()
    ygrid = np.array(dict)
    xgrid = np.arange(len(pred))
    
    xx = []
    last_ph = ''
    for i in ph_pred:
        if i==last_ph:
            xx.append('')
        else:
            xx.append(i)
            last_ph = i
    # print(xgrid)
    ax.pcolormesh(xgrid, ygrid, tf.transpose(pred), shading = 'nearest')
    plt.xticks(fontsize=8)
    

    plt.savefig(f'{time.strftime("%Y%m%d%H%M%S")}.png', dpi=400)

if mode == 'train':
    token_layer, ds = init()
    # vocab = token_layer.get_vocabulary()

    masked_token = tf.cast(token_layer('sil')[0], dtype = tf.int64)

    train_ds, val_ds, test_ds = init_ds()
    model = PPG_CNN(tokenizer = token_layer)
    train_model(masked_token = masked_token)

elif mode == 'eval':
    token_layer, ds = init()
    train_ds, val_ds, test_ds = init_ds()
    eval_model(token_layer = token_layer)

