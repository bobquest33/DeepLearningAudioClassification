from time import sleep

import tflearn
import numpy as np
from tflearn.layers.merge_ops import merge
import os
from audio_reader import wav_batch_generator, random_pick_to_test_dataset, put_back_test_dataset

test_data = './test_data'
test_vocal_data = os.path.join(test_data, './vocal')
test_non_vocal_data = os.path.join(test_data, './non_vocal')
train_vocal_dir = './vocal'
train_non_vocal_dir = './non_vocal'


batch_size = 1400
test_batch_size = 150
sample_size = 16000

n_classes = 2

tflearn.init_graph(num_cores=4, gpu_memory_fraction=0.5)

inp = tflearn.input_data(shape=[None, sample_size, 1], name='input')
stride1 = tflearn.conv_1d(inp, 4, 256, 160, activation='relu', regularizer='L2', bias=False)
stride_pool1 = tflearn.max_pool_1d(stride1, 4)

stride2 = tflearn.conv_1d(inp, 4, 512, 320, activation='relu', regularizer='L2', bias=False)
stride_pool2 = tflearn.max_pool_1d(stride2, 2)
stride_out = merge([stride_pool1, stride_pool2], mode='concat', axis=2)
net = tflearn.conv_1d(stride_out, 32, 8, activation='relu', regularizer='L2', bias=False)

net = tflearn.max_pool_1d(net, 4)
net = tflearn.conv_1d(net, 32, 8, activation='relu', regularizer='L2', bias=False)

net = tflearn.fully_connected(net, 50, bias=False)
net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net, n_classes, activation='softmax', bias=False)
net = tflearn.regression(net, optimizer='adam', learning_rate=0.00001,
                         loss='categorical_crossentropy', name='target')
model = tflearn.DNN(net, tensorboard_verbose=2, checkpoint_path='./log', max_checkpoints=1)
try:
    put_back_test_dataset(test_data, train_vocal_dir, train_non_vocal_dir)
    random_pick_to_test_dataset(train_vocal_dir, train_non_vocal_dir, test_data)
    batch = wav_batch_generator(train_vocal_dir, train_non_vocal_dir, sample_size=sample_size, batch_size=batch_size)
    X, Y = next(batch)
    X = np.reshape(X, [batch_size, sample_size, 1])

    test_batch = wav_batch_generator(test_vocal_data, test_non_vocal_data, sample_size=sample_size,
                                     batch_size=test_batch_size)
    test_X, test_Y = next(test_batch)
    test_X = np.reshape(test_X, [test_batch_size, sample_size, 1])

    model.fit(X, Y, n_epoch=120, show_metric=True, batch_size=80, snapshot_step=30,
              validation_set=(test_X, test_Y)
              , run_id='4plus4-all-no-bias{}'.format(1))
    model.save('my_model.tflearn')
except KeyboardInterrupt:
    model.save('my_model.tflearn')
