import tensorflow as tf

from X_Vector_Layer import XVectorLayer
import os

from datetime import date

ds_dir = os.path.expanduser('~/tmp/dataset_05_09_mel_80')

ds = tf.data.Dataset.load(ds_dir)

sp = tf.keras.layers.TextVectorization()

sp_ds = ds.map(lambda mel, spk: spk)

sp.adapt(sp_ds)

n_mels = 80
ds = ds.map(lambda mel, spk: (tf.reshape(mel, [-1, n_mels]), sp(spk)[0]))

num_speaker = len(sp.get_vocabulary(include_special_tokens = True))

model = XVectorLayer(num_speaker = num_speaker, if_output_x_vector = False)

# model = tf.keras.Sequential()
# model.add(xV)

def split_dataset(ds, train = 0.85, val = 0.1, test = 0.05, batch = 1):
    ds_size = len(ds)
    ds = ds.shuffle(buffer_size = 10000)

    train_size = int(train*ds_size)
    val_size = int(val*ds_size)

    train_ds = ds.take(train_size).batch(batch)
    val_ds = ds.skip(train_size).take(val_size).batch(batch)
    test_ds = ds.skip(train_size).skip(val_size).batch(batch)

    return train_ds, val_ds, test_ds

train_ds, val_ds, test_ds = split_dataset(ds, batch = 1)

def train_model():
    optm = tf.keras.optimizers.Adam()
    model.compile(optimizer = optm, loss = 'sparse_categorical_crossentropy', metrics = 'sparse_categorical_accuracy')

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath = './checkpoints_'+f'{date.today().strftime("%b_%d_%Y")}'+'/ckp.{epoch:02d}',
        save_weights_only = True,
        save_best_only = True,
        verbose = 1)

    class NaNCallback(tf.keras.callbacks.Callback):
        def on_batch_end(self, batch, logs=None):
            print(
                f"Up to batch {batch}, the average loss is {logs['loss']:7.2f}")
            
                
    
    history = model.fit(
        train_ds,
        epochs = 100,
        validation_data = val_ds,
        callbacks = [checkpoint_callback])

    print('Evaluating trained model')
    model.evaluate(test_ds)

    return history

history = None

if __name__ == '__main__':
    history = train_model()


