import os

import h5py

from audio_reader import *

sample_size = 16000
sample_rate = 16000
testing_percent = 0.1
n_classes = 2
batch_size = 100
test_data = './mp3/test_data'
test_vocal_data = os.path.join(test_data, './vocal')
test_non_vocal_data = os.path.join(test_data, './non_vocal')
train_vocal_dir = './mp3/vocal'
train_non_vocal_dir = './mp3/non_vocal'

h5py.File('data.h5', 'w')
h5 = h5py.File('data.h5', 'w')

train_batch_size = int(batch_size * (1 - testing_percent))
test_batch_size = int(batch_size * testing_percent)

x_dataset = h5.create_dataset('X', (0, sample_size, 1), maxshape=(None, sample_size, 1))
y_dataset = h5.create_dataset('Y', (0, n_classes), maxshape=(None, n_classes))
test_x_dataset = h5.create_dataset('test_X', (0, sample_size, 1), maxshape=(None, sample_size, 1))
test_y_dataset = h5.create_dataset('test_Y', (0, n_classes), maxshape=(None, n_classes))
batch_generator = BatchGenerator(train_vocal_dir, train_non_vocal_dir)
batch = batch_generator.wav_batch_generator(batch_size, sample_size=sample_size)

while not batch_generator.ended:
    x_dataset.resize(x_dataset.shape[0] + train_batch_size, axis=0)
    y_dataset.resize(y_dataset.shape[0] + train_batch_size, axis=0)
    test_x_dataset.resize(test_x_dataset.shape[0] + test_batch_size, axis=0)
    test_y_dataset.resize(test_y_dataset.shape[0] + test_batch_size, axis=0)

    X_DATA, Y_DATA = next(batch)
    X_DATA = np.reshape(X_DATA, [batch_size, sample_size, 1])

    test_indexes = set(np.random.choice(X_DATA.shape[0], test_batch_size, replace=False))
    indexes = set(np.arange(X_DATA.shape[0])) - test_indexes

    test_X = X_DATA[list(test_indexes)]
    X = X_DATA[list(indexes)]

    test_Y = Y_DATA[list(test_indexes)]
    Y = Y_DATA[list(indexes)]

    x_dataset[-train_batch_size:] = X
    y_dataset[-train_batch_size:] = Y
    test_x_dataset[-test_batch_size:] = test_X
    test_y_dataset[-test_batch_size:] = test_Y

h5.close()
