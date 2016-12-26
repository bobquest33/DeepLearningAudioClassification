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
        self.learning_rate = 0.0006
        self.training_iters = 800000
        self.batch_size = 40
        self.display_step = 5
        self.validate_step = 5
        self.test_step = 5
        self.checkpoint_every = 400
        # Network Parameters
        self.n_input = 32000  # MNIST data input (img shape: 28*28)
        self.n_classes = 2  # MNIST total classes (0-9 digits)
        self.dropout = 0.75  # Dropout, probability to keep units
        self.logdir = './log'

        # tf Graph input
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.y = tf.placeholder(tf.float32, [None, self.n_classes])
        self.keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

        # Store layers weight & bias
        self.weights = {
            # 5 conv, 1 input, 32 outputs
            'st1': tf.Variable(tf.random_normal([800, 1, 192])),
            'wc1': tf.Variable(tf.random_normal([30, 192, 192])),
            'wc2': tf.Variable(tf.random_normal([30, 192, 256])),
            'wc3': tf.Variable(tf.random_normal([30, 256, 512])),
            # fully connected, 7*7*64 inputs, 1024 outputs
            'wd1': tf.Variable(tf.random_normal([8 * 512, 100])),
            # 1024 inputs, 10 outputs (class prediction)
            'out': tf.Variable(tf.random_normal([100, self.n_classes]))
        }

        self.biases = {
            'st1': tf.Variable(tf.random_normal([192])),
            'bc1': tf.Variable(tf.random_normal([192])),
            'bc2': tf.Variable(tf.random_normal([256])),
            'bc3': tf.Variable(tf.random_normal([512])),
            'bd1': tf.Variable(tf.random_normal([100])),
            'bd2': tf.Variable(tf.random_normal([50])),
            'out': tf.Variable(tf.random_normal([self.n_classes]))
        }

    # Create some wrappers for simplicity
    def conv1d(self, x, W, b, strides=1):
        # Conv1D wrapper, with bias and relu activation
        x = tf.nn.conv1d(x, W, stride=strides, padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def maxpool2d(self, x, channels, k=3):
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
        stride_conv = self.conv1d(x, weights['st1'], biases['st1'], strides=160)

        # Convolution Layer
        conv1 = self.conv1d(stride_conv, weights['wc1'], biases['bc1'])
        conv1 = self.maxpool2d(conv1, weights['wc1'].get_shape().as_list()[-1])

        # Convolution Layer
        conv2 = self.conv1d(conv1, weights['wc2'], biases['bc2'])
        conv2 = self.maxpool2d(conv2, weights['wc2'].get_shape().as_list()[-1])

        # Convolution Layer
        conv3 = self.conv1d(conv2, weights['wc3'], biases['bc3'])
        conv3 = self.maxpool2d(conv3, weights['wc3'].get_shape().as_list()[-1])

        # Fully connected layer
        fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        fc1 = tf.nn.dropout(fc1, dropout)

        # Output, class prediction
        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
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
        pred = self.conv_net(self.x, self.weights, self.biases, self.keep_prob)
        # Define loss and optimizer
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(pred, self.y)
        cost = tf.reduce_mean(cross_entropy)
        tf.scalar_summary('cost', cost)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)

        # Evaluate model
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(self.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.scalar_summary('acc', accuracy)
        writer = tf.train.SummaryWriter('board')
        writer.add_graph(tf.get_default_graph())
        run_metadata = tf.RunMetadata()
        summaries = tf.merge_all_summaries()
        test_accuracy_summary = tf.scalar_summary('test', accuracy)

        # Initializing the variables
        init = tf.initialize_all_variables()
        audio_reader = AudioReader(sample_size=self.n_input)
        audio_reader.get_all_batches()
        audio_reader.tied_batches()

        test_audio_reader = AudioReader(sample_size=self.n_input,
                                        vocal_audio_directory='./test_data/vocal',
                                        non_vocal_audio_directory='./test_data/non_vocal')
        test_audio_reader.get_all_batches()
        test_audio_reader.tied_batches()
        # audio_reader = WrongAudioReader(sample_size=self.n_input)
        # audio_reader.get_all(20)
        # Launch the graph
        with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
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
            FilterDisplay.show_weight(sess.run(self.weights['st1']))
            sleep(1000)
            while step < self.training_iters:

                audio_sample, audio_label, _ = audio_reader.next_batch(self.batch_size)

                # print(sess.run([pred], feed_dict={self.x: audio_sample, self.y: audio_label,
                #                                   self.keep_prob: self.dropout})[0].shape)
                # print(audio_sample[0])
                # sleep(1000)
                summary, _, _ = sess.run([summaries, optimizer, accuracy],
                                         feed_dict={self.x: audio_sample, self.y: audio_label,
                                                    self.keep_prob: self.dropout})

                writer.add_summary(summary, step)
                if step % self.display_step == 0:
                    # Calculate batch loss and accuracy
                    # print(sess.run(correct_pred, feed_dict={x: audio_sample, y: audio_label,
                    #                                         keep_prob: 1.}))
                    loss, acc = sess.run([cost, accuracy], feed_dict={self.x: audio_sample,
                                                                      self.y: audio_label,
                                                                      self.keep_prob: 1.})

                    print("Iter " + str(step * self.batch_size) + ", Minibatch Loss= " + \
                          "{:.6f}".format(loss) + ", Training Accuracy= " + \
                          "{:.5f}".format(acc))

                if step % self.validate_step == 0:
                    audio_sample, audio_label, _ = audio_reader.pick_random(self.batch_size)
                    accuracy_val = sess.run(accuracy, feed_dict={self.x: audio_sample,
                                                                 self.y: audio_label,
                                                                 self.keep_prob: 1.})
                    print("Validation Accuracy:", accuracy_val)
                    # audio_sample_begin += n_input
                    # step += 1

                if step % self.test_step == 0:
                    audio_sample, audio_label, _ = test_audio_reader.pick_random(self.batch_size)
                    accuracy_val, test_accuracy_summ = sess.run([accuracy, test_accuracy_summary],
                                                                feed_dict={self.x: audio_sample,
                                                                           self.y: audio_label,
                                                                           self.keep_prob: 1.})
                    print("Testing Accuracy:", accuracy_val)
                    writer.add_summary(test_accuracy_summ, step)
                    # audio_sample_begin += n_input
                    # step += 1
                if step % self.checkpoint_every == 0:
                    self.save(saver, sess, self.logdir, step)
                step += 1

        print("Optimization Finished!")


AudioClassification().run()
