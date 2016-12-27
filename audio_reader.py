import fnmatch
import os
from random import shuffle, random
from time import sleep
import numpy as np
import librosa
import random
import tensorflow as tf
import threading

from definitions import ROOT_DIR

MFCC_N = 20


def find_files(directory, pattern='*.wav'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files


def load_generic_audio(directory, sample_rate):
    files = find_files(directory)
    # files_count = len(files)
    for filename in files:
        audio, sr = librosa.load(filename, sr=sample_rate, mono=True)
        yield audio


class AudioReader:
    def __init__(self,
                 coord,
                 queue_size=256,
                 min_after_dequeue=0,
                 n_classes=2,
                 sample_rate=16000,
                 sample_size=32000,
                 vocal_audio_directory='./mp3/vocal',
                 non_vocal_audio_directory='./mp3/non_vocal'):
        self.coord = coord
        self.queue_size = queue_size
        self.sample_rate = sample_rate
        self.sample_size = sample_size
        self.n_classes = n_classes
        self.x = tf.placeholder(tf.float32, shape=None)
        self.y = tf.placeholder(tf.float32, shape=None)
        self.queue = tf.RandomShuffleQueue(queue_size, min_after_dequeue, ['float32', 'float32'],
                                           shapes=[(self.sample_size, 1), (self.n_classes, 1)])
        self.enqueue = self.queue.enqueue([self.x, self.y])
        self.vocal_audio_directory = os.path.join(ROOT_DIR, vocal_audio_directory)
        self.non_vocal_audio_directory = os.path.join(ROOT_DIR, non_vocal_audio_directory)
        # self.audios = self.read_dir(vocal_audio_directory, self.sample_rate, is_vocal=True)
        # self.audios.extend(self.read_dir(non_vocal_audio_directory, self.sample_rate))
        # shuffle(self.audios)
        self.current_audio = None
        self.current_audio_label = None
        self.current_audio_index = 0
        self.current_audio_sample = 0
        self.samples = []
        self.labels = []
        self.numbers = []
        self.tuples = []
        self.current_batch_index = 0
        self.roll = False

    def dequeue(self, num_elements):
        output = self.queue.dequeue_many(num_elements)
        return output

    def thread_main(self, sess):
        print('start')
        stop = False
        while not stop:
            vocal_iterator = load_generic_audio(self.vocal_audio_directory, self.sample_rate)
            non_vocal_iterator = load_generic_audio(self.non_vocal_audio_directory, self.sample_rate)
            for vocal_audio, non_vocal_audio in zip(vocal_iterator, non_vocal_iterator):
                if self.coord.should_stop():
                    stop = True
                    break
                vocal_buffer_ = np.array(vocal_audio)
                non_vocal_buffer_ = np.array(non_vocal_audio)
                while len(vocal_buffer_) > self.sample_size:
                    piece = np.reshape(vocal_buffer_[:self.sample_size], [-1, 1])
                    sess.run(self.enqueue, feed_dict={self.x: piece, self.y: np.array([[0], [1]])})
                    vocal_buffer_ = vocal_buffer_[self.sample_size:]
                while len(non_vocal_buffer_) > self.sample_size:
                    piece = np.reshape(non_vocal_buffer_[:self.sample_size], [-1, 1])
                    sess.run(self.enqueue, feed_dict={self.x: piece, self.y: np.array([[1], [0]])})
                    non_vocal_buffer_ = non_vocal_buffer_[self.sample_size:]
        self.coord.request_stop()

    def start_thread(self, sess):
        thread = threading.Thread(target=self.thread_main, args=(sess,))
        thread.daemon = True
        thread.start()

    def get_all_batches(self):
        self.tuples = []
        index = 1
        for current_audio, is_vocal in self.audios:
            current_audio_sample = 0
            label = [0, 1] if is_vocal == 1 else [1, 0]
            while current_audio_sample + self.sample_size < current_audio.shape[0]:
                sample = current_audio[current_audio_sample:current_audio_sample + self.sample_size]
                self.tuples.append((sample, label, index))
                current_audio_sample += self.sample_size
                index += 1

    def tied_batches(self, is_shuffle=True):
        if is_shuffle:
            shuffle(self.tuples)
        print("total sample number is {}".format(len(self.tuples)))
        for sample, label, index in self.tuples:
            self.samples.append(sample)
            self.labels.append(label)
            self.numbers.append(index)

    def batch_is_rolling_back(self):
        return self.roll

    def next_batch(self, batch_size):
        if self.current_batch_index + batch_size > len(self.samples):
            self.current_batch_index = 0
            self.roll = True
        samples_batch = self.samples[self.current_batch_index:self.current_batch_index + batch_size]
        labels_batch = self.labels[self.current_batch_index:self.current_batch_index + batch_size]
        numbers_batch = self.numbers[self.current_batch_index:self.current_batch_index + batch_size]

        samples_batch = np.array(samples_batch)
        self.current_batch_index += batch_size
        return samples_batch, labels_batch, numbers_batch

    def pick_random(self, pick_nums):
        random_list = random.sample(list(np.arange(len(self.samples))), pick_nums)
        samples_batch = np.array([self.samples[random_num] for random_num in random_list])
        labels_batch = np.array([self.labels[random_num] for random_num in random_list])
        numbers_batch = np.array([self.numbers[random_num] for random_num in random_list])
        return samples_batch, labels_batch, numbers_batch

    def read_dir(self, dir, sample_rate, is_vocal=False):
        audio_files = []
        label = 1 if is_vocal else 0
        for audio in load_generic_audio(dir, sample_rate):
            audio_files.append((audio, label))
        return audio_files


        # AudioReader(sample_size=32000,
        #             vocal_audio_directory='./mp3/vocal',
        #             non_vocal_audio_directory='./mp3/non_vocal')
        #
