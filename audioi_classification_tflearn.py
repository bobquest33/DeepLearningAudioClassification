from time import sleep

import tflearn
import numpy as np
from audio_reader import wav_batch_generator

batch_size = 3000
test_batch_size = 30
sample_size = 64000

test_batch = wav_batch_generator('./mp3/test_data/vocal', './mp3/test_data/non_vocal', sample_size=sample_size,
                                 batch_size=test_batch_size)
test_X, test_Y = next(test_batch)
test_X = np.reshape(test_X, [test_batch_size, sample_size, 1])

n_classes = 2

tflearn.init_graph(num_cores=4, gpu_memory_fraction=0.5)

net = tflearn.input_data(shape=[None, sample_size, 1], name='input')
net = tflearn.conv_1d(net, 32, 512, 512, activation='relu', regularizer='L2')
net = tflearn.max_pool_1d(net, 4)
net = tflearn.conv_1d(net, 32, 8, activation='relu', regularizer='L2')
net = tflearn.max_pool_1d(net, 4)
net = tflearn.conv_1d(net, 32, 8, activation='relu', regularizer='L2')
net = tflearn.fully_connected(net, 100)
net = tflearn.dropout(net, 0.8)
net = tflearn.fully_connected(net, 50)
net = tflearn.dropout(net, 0.8)
net = tflearn.fully_connected(net, n_classes, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.00001,
                         loss='categorical_crossentropy', name='target')
model = tflearn.DNN(net, tensorboard_verbose=1, checkpoint_path='./log', max_checkpoints=1)
# try:
batch = wav_batch_generator('./mp3/vocal', './mp3/non_vocal', sample_size=sample_size, batch_size=batch_size)
X, Y = next(batch)
X = np.reshape(X, [batch_size, sample_size, 1])
model.fit(X, Y, n_epoch=100, show_metric=True, batch_size=8, snapshot_step=100,
          validation_set=(test_X, test_Y)
          , run_id='magnatag-64')
model.save('my_model.tflearn')
# except KeyboardInterrupt:
#     model.save('my_model.tflearn')
