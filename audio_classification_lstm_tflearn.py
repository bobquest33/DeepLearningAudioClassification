from time import sleep

import tflearn
import numpy as np
from tflearn.layers.merge_ops import merge

from audio_reader import wav_batch_generator, random_pick_to_test_dataset, put_back_test_dataset

batch_size = 280
test_batch_size = 30
sample_size = 4800

n_classes = 2

tflearn.init_graph(num_cores=4, gpu_memory_fraction=0.5)

inp = tflearn.input_data(shape=[None, sample_size, 1], name='input')
stride = tflearn.conv_1d(inp, 32, 1024, 1024, activation='relu', regularizer='L2')
net = tflearn.lstm(stride, 128, dropout=0.8)
net = tflearn.fully_connected(net, n_classes, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.00001,
                         loss='categorical_crossentropy', name='target')
model = tflearn.DNN(net, tensorboard_verbose=1, checkpoint_path='./log', max_checkpoints=1)
try:
    put_back_test_dataset('./test_data', './vocal', './non_vocal')
    random_pick_to_test_dataset('./vocal', './non_vocal', './test_data')
    batch = wav_batch_generator('./vocal', './non_vocal', sample_size=sample_size, batch_size=batch_size)
    X, Y = next(batch)
    X = np.reshape(X, [batch_size, sample_size, 1])

    test_batch = wav_batch_generator('./test_data/vocal', './test_data/non_vocal', sample_size=sample_size,
                                     batch_size=test_batch_size)
    test_X, test_Y = next(test_batch)
    test_X = np.reshape(test_X, [test_batch_size, sample_size, 1])
    model.fit(X, Y, n_epoch=70, show_metric=True, batch_size=10, snapshot_step=30,
              validation_set=(test_X, test_Y)
              , run_id='youlstm-{}'.format(0))
    model.save('my_model.tflearn')
except KeyboardInterrupt:
    model.save('my_model.tflearn')
