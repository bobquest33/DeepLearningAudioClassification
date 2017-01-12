import os

import h5py

from audio_reader import *

sample_size = 16000
batch_size = 10000
test_data = './mp3/test_data'
test_vocal_data = os.path.join(test_data, './vocal')
test_non_vocal_data = os.path.join(test_data, './non_vocal')
train_vocal_dir = './mp3/vocal'
train_non_vocal_dir = './mp3/non_vocal'

h5py.File('data.h5', 'w')
h5 = h5py.File('data.h5', 'w')
batch_generator = BatchGenerator(train_vocal_dir, train_non_vocal_dir)
batch = batch_generator.wav_batch_generator(batch_size, s
for index in range(0, 72600, 1000):
    pass
h5.create_dataset('X', data=X)
h5.create_dataset('y', data=Y)
h5.close()
