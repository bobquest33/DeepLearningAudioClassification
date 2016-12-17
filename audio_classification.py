'''
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf

# Import MNIST data
# from tensorflow.examples.tutorials.mnist import input_data
from audio_reader import *
import sys

# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.001
training_iters = 400
batch_size = 100
display_step = 5
validate_step = 1
checkpoint_every = 400
# Network Parameters
n_input = 400  # MNIST data input (img shape: 28*28)
sample_size = 20
mfcc_n = 20
n_classes = 2  # MNIST total classes (0-9 digits)
dropout = 0.75  # Dropout, probability to keep units
logdir = './log'

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv1D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return x


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[batch_size, sample_size, mfcc_n, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)
    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir), end="")
    sys.stdout.flush()

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print(' Done.')


def load(saver, sess, logdir):
    print("Trying to restore saved checkpoints from {} ...".format(logdir),
          end="")

    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt:
        print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
        global_step = int(ckpt.model_checkpoint_path
                          .split('/')[-1]
                          .split('-')[-1])
        print("  Global step was: {}".format(global_step))
        print("  Restoring...", end="")
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(" Done.")
        return global_step
    else:
        print(" No checkpoint found.")
        return None


# Store layers weight & bias
weights = {
    # 5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([int(sample_size / 4) * int(mfcc_n / 4) * 64, 512])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([512, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([512])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)
# Define loss and optimizer
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(pred, y)
cost = tf.reduce_mean(cross_entropy)
cost_summary = tf.scalar_summary('cost', cost)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

accuracy_summary = tf.scalar_summary('acc', accuracy)
validation_summary = tf.scalar_summary('validate', accuracy)

writer = tf.train.SummaryWriter('board')

writer.add_graph(tf.get_default_graph())
run_metadata = tf.RunMetadata()
summaries = tf.merge_all_summaries()

# Initializing the variables

init = tf.initialize_all_variables()
audio_reader = AudioReader(sample_size=sample_size)
audio_reader.get_all_batches()
audio_reader.shuffle_batches()

validate_audio_reader = AudioReader(sample_size=sample_size,
                                    vocal_audio_directory='./validate_vocal',
                                    non_vocal_audio_directory='./validate_no_vocal')
validate_audio_reader.get_all_batches()
validate_audio_reader.shuffle_batches()

# Launch the graph
with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    saver = tf.train.Saver(var_list=tf.trainable_variables())
    try:
        step = load(saver, sess, logdir)
        if step == None:
            step = 1
    except:
        print("Something went wrong while restoring checkpoint. "
              "We will terminate training to avoid accidentally overwriting "
              "the previous model.")
        raise

    while step <= training_iters:
        audio_sample, audio_label, _ = audio_reader.next_batch(batch_size)
        summary, cost_summ, _, _ = sess.run([accuracy_summary, cost_summary, optimizer, accuracy],
                                            feed_dict={x: audio_sample, y: audio_label,
                                                       keep_prob: dropout})
        writer.add_summary(summary, step)
        writer.add_summary(cost_summ, step)
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            # print(sess.run(correct_pred, feed_dict={x: audio_sample, y: audio_label,
            #                                         keep_prob: 1.}))
            loss, acc = sess.run([cost, accuracy], feed_dict={x: audio_sample,
                                                              y: audio_label,
                                                              keep_prob: 1.})

            # print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
            #       "{:.6f}".format(loss) + ", Training Accuracy= " + \
            #       "{:.5f}".format(acc))

        if step % validate_step == 0:
            audio_sample, audio_label, _ = validate_audio_reader.next_batch(batch_size)
            summary, validate_acc = sess.run([validation_summary, accuracy], feed_dict={x: audio_sample,
                                                                                        y: audio_label,
                                                                                        keep_prob: 1.})
            writer.add_summary(summary, step)
            print("Testing Accuracy:", validate_acc)
        if step % checkpoint_every == 0:
            save(saver, sess, logdir, step)
        step += 1

print("Optimization Finished!")
