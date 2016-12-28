import tflearn
import numpy as np
from audio_reader import wav_batch_generator

batch_size = 256
sample_size = 32000
batch = wav_batch_generator('./vocal', './non_vocal', sample_size=sample_size, batch_size=batch_size)
X, Y = next(batch)
X = np.reshape(X, [batch_size, sample_size, 1])
n_classes = 2

tflearn.init_graph(num_cores=8)

# net = tflearn.input_data(shape=[None, 48000])
# net = tflearn.fully_connected(net, 64)
# net = tflearn.dropout(net, 0.5)
# net = tflearn.fully_connected(net, n_classes, activation='softmax')
# net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy')

net = tflearn.input_data(shape=[None, 32000, 1], name='input')
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
net = tflearn.regression(net, optimizer='adam', learning_rate=0.01,
                         loss='categorical_crossentropy', name='target')
model = tflearn.DNN(net, tensorboard_verbose=2)
model.fit(X, Y, n_epoch=200, show_metric=True)
