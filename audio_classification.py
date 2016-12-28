'''
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
import numpy

from utils.filter_display import FilterDisplay

numpy.set_printoptions(threshold=numpy.nan)
# Import MNIST data
# from tensorflow.examples.tutorials.mnist import input_data
from audio_reader import *
import sys

# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
from wrong_audio_reader import WrongAudioReader


class AudioClassification:
    def __init__(self):
        self.learning_rate = 0.001
        self.training_iters = 800000
        self.batch_size = 10
        self.display_step = 5
        self.validate_step = 5
        self.test_step = 5
        self.checkpoint_every = 400
        # Network Parameters
        self.n_input = 48000  # MNIST data input (img shape: 28*28)
        self.n_classes = 2  # MNIST total classes (0-9 digits)
        self.dropout = 0.75  # Dropout, probability to keep units
        self.logdir = './log'
        self.coord = tf.train.Coordinator()
        # tf Graph input
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.y = tf.placeholder(tf.float32, [None, self.n_classes])
        self.keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

        # Store layers weight & bias
        self.weights = {
            # 5 conv, 1 input, 32 outputs
            'st1': tf.Variable(tf.random_normal([512, 1, 32])),
            'wc1': tf.Variable(tf.random_normal([8, 32, 32])),
            'wc2': tf.Variable(tf.random_normal([8, 32, 32])),
            # fully connected, 7*7*64 inputs, 1024 outputs
            'wd1': tf.Variable(tf.random_normal([6 * 32, 100])),
            'wd2': tf.Variable(tf.random_normal([100, 50])),
            # 1024 inputs, 10 outputs (class prediction)
            'out': tf.Variable(tf.random_normal([50, self.n_classes]))
        }

        self.biases = {
            'st1': tf.Variable(tf.random_normal([32])),
            'bc1': tf.Variable(tf.random_normal([32])),
            'bc2': tf.Variable(tf.random_normal([32])),
            'bd1': tf.Variable(tf.random_normal([100])),
            'bd2': tf.Variable(tf.random_normal([50])),
            'out': tf.Variable(tf.random_normal([self.n_classes]))
        }

        self.filter_display = FilterDisplay()

    # Create some wrappers for simplicity
    def conv1d(self, x, W, b, strides=1):
        # Conv1D wrapper, with bias and relu activation
        x = tf.nn.conv1d(x, W, stride=strides, padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def maxpool2d(self, x, channels, k=4):
        # MaxPool2D wrapper
        x = tf.reshape(x, shape=[self.batch_size, 1, -1, channels])
        out = tf.nn.max_pool(x, ksize=[1, 1, k, 1], strides=[1, 1, k, 1],
                             padding='SAME')
        return tf.reshape(out, shape=[self.batch_size, -1, channels])

    # Create model
    def conv_net(self, x, weights, biases, dropout):
        # Reshape input picture
        x = tf.reshape(x, shape=[-1, self.n_input, 1])

        # Strided Layer
        stride_conv = self.conv1d(x, weights['st1'], biases['st1'], strides=512)

        # Convolution Layer
        conv1 = self.conv1d(stride_conv, weights['wc1'], biases['bc1'])
        conv1 = self.maxpool2d(conv1, weights['wc1'].get_shape().as_list()[-1])

        # Convolution Layer
        conv2 = self.conv1d(conv1, weights['wc2'], biases['bc2'])
        conv2 = self.maxpool2d(conv2, weights['wc2'].get_shape().as_list()[-1])


        # Fully connected layer
        fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)

        fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
        fc2 = tf.nn.dropout(fc2, dropout)

        # Output, class prediction
        out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
        return out

    def save(self, saver, sess, logdir, step):
        model_name = 'model.ckpt'
        checkpoint_path = os.path.join(logdir, model_name)
        print('Storing checkpoint to {} ...'.format(logdir), end="")
        sys.stdout.flush()

        if not os.path.exists(logdir):
            os.makedirs(logdir)

        saver.save(sess, checkpoint_path, global_step=step)
        print(' Done.')

    def load(self, saver, sess, logdir):
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

    # Construct model

    def run(self):
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))

        audio_reader = AudioReader(coord=self.coord, sample_size=self.n_input)

        test_audio_reader = AudioReader(coord=self.coord,
                                        sample_size=self.n_input,
                                        queue_size=42,
                                        vocal_audio_directory='./test_data/vocal',
                                        non_vocal_audio_directory='./test_data/non_vocal')
        audio_reader.start_thread(sess)
        test_audio_reader.start_thread(sess)

        audio_batch, label_batch = audio_reader.dequeue(self.batch_size)
        label_batch = tf.reshape(label_batch, shape=[self.batch_size, self.n_classes])

        test_audio_batch, test_label_batch = test_audio_reader.dequeue(self.batch_size)
        test_label_batch = tf.reshape(label_batch, shape=[self.batch_size, self.n_classes])

        pred = self.conv_net(audio_batch, self.weights, self.biases, self.keep_prob)
        test_pred = self.conv_net(test_audio_batch, self.weights, self.biases, self.keep_prob)
        # Define loss and optimizer
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(pred, label_batch)
        cost = tf.reduce_mean(cross_entropy)
        tf.scalar_summary('cost', cost)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)

        # Evaluate model
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(label_batch, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.scalar_summary('acc', accuracy)

        test_correct_pred = tf.equal(tf.argmax(test_pred, 1), tf.argmax(test_label_batch, 1))
        test_accuracy = tf.reduce_mean(tf.cast(test_correct_pred, tf.float32))

        writer = tf.train.SummaryWriter('board')
        writer.add_graph(tf.get_default_graph())
        run_metadata = tf.RunMetadata()
        summaries = tf.merge_all_summaries()
        # test_accuracy_summary = tf.scalar_summary('test', accuracy)

        # Initializing the variables
        init = tf.initialize_all_variables()

        # Launch the graph


        sess.run(init)

        step = 1
        # Keep training until reach max iterations
        saver = tf.train.Saver(var_list=tf.trainable_variables())
        try:
            step = self.load(saver, sess, self.logdir)
            if step == None:
                step = 1
        except:
            print("Something went wrong while restoring checkpoint. "
                  "We will terminate training to avoid accidentally overwriting "
                  "the previous model.")
            raise
        while step < self.training_iters:
            # self.filter_display.show_weight(sess.run(self.weights['st1']))
            # audio_sample, audio_label, _ = audio_reader.next_batch(self.batch_size)

            # print(sess.run([pred], feed_dict={self.x: audio_sample, self.y: audio_label,
            #                                   self.keep_prob: self.dropout})[0].shape)
            # print(audio_sample[0])
            # sleep(1000)
            print("current step: {}".format(step))
            summary, _, acc = sess.run([summaries, optimizer, accuracy],
                                       feed_dict={self.keep_prob: self.dropout})
            print("Iter " + str(step * self.batch_size) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
            writer.add_summary(summary, step)
            # if step % self.display_step == 0:
            #     # Calculate batch loss and accuracy
            #     # print(sess.run(correct_pred, feed_dict={x: audio_sample, y: audio_label,
            #     #                                         keep_prob: 1.}))
            #     loss, acc = sess.run([cost, accuracy], feed_dict={self.keep_prob: 1.})
            #
            #     print("Iter " + str(step * self.batch_size) + ", Minibatch Loss= " + \
            #           "{:.6f}".format(loss) + ", Training Accuracy= " + \
            #           "{:.5f}".format(acc))

            # if step % self.validate_step == 0:
            #     audio_sample, audio_label, _ = audio_reader.pick_random(self.batch_size)
            #     accuracy_val = sess.run(accuracy, feed_dict={self.x: audio_sample,
            #                                                  self.y: audio_label,
            #                                                  self.keep_prob: 1.})
            #     print("Validation Accuracy:", accuracy_val)
            #     # audio_sample_begin += n_input
            #     # step += 1

            if step % self.test_step == 0:
                # audio_sample, audio_label, _ = test_audio_reader.pick_random(self.batch_size)
                accuracy_val = sess.run(test_accuracy,
                                        feed_dict={self.keep_prob: 1.})
                print("Testing Accuracy:", accuracy_val)
                # writer.add_summary(test_accuracy_summ, step)
                # audio_sample_begin += n_input
                # step += 1
            if step % self.checkpoint_every == 0:
                self.save(saver, sess, self.logdir, step)
            step += 1

        print("Optimization Finished!")


AudioClassification().run()
