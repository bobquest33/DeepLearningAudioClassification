import fnmatch
import os
import wave
from random import shuffle, random
from time import sleep
import numpy as np
import librosa
import random
import tensorflow as tf
import threading
import shutil

from definitions import ROOT_DIR

MFCC_N = 20
CHUNK = 36000


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
        yield audio, filename


def tag_and_rename_files(vocal_directory, non_vocal_directory):
    vocal_files = os.listdir(vocal_directory)
    non_vocal_files = os.listdir(non_vocal_directory)
    for file in vocal_files:
        new_file = os.path.join(vocal_directory, "v_{}".format(file))
        file = os.path.join(vocal_directory, file)
        os.rename(file, new_file)
    for file in non_vocal_files:
        new_file = os.path.join(non_vocal_directory, "n_{}".format(file))
        file = os.path.join(non_vocal_directory, file)
        os.rename(file, new_file)


def load_wav_file(name):
    f = wave.open(name, "rb")
    # print("loading %s"%name)
    chunk = []
    data0 = f.readframes(CHUNK)
    while data0:  # f.getnframes()
        # data=numpy.fromstring(data0, dtype='float32')
        # data = numpy.fromstring(data0, dtype='uint16')
        data = np.fromstring(data0, dtype='uint16')
        data = (data + 32768) / 65536.  # 0-1 for Better convergence
        # chunks.append(data)
        chunk.extend(data)
        data0 = f.readframes(CHUNK)
    # finally trim:
    # chunk.extend(np.zeros(CHUNK * 2 - len(chunk)))  # fill with padding 0's
    # print("%s loaded"%name)
    return chunk


def wav_batch_generator(vocal_directory, non_vocal_directory, batch_size=10, sample_size=32000, sample_rate=16000):
    files = find_files(vocal_directory)
    files.extend(find_files(non_vocal_directory))
    batch_waves = []
    batch_labels = []
    while True:
        shuffle(files)
        file_count = 0
        for filename in files:
            audio = load_wav_file(filename)
            filename = os.path.basename(filename)
            label_eye = 1 if filename[0] == 'n' else 0
            label = np.eye(2)[label_eye]
            while len(audio) >= sample_size:
                waves = audio[:sample_size]
                batch_waves.append(waves)
                batch_labels.append(label)
                if len(batch_waves) >= batch_size:
                    yield batch_waves, batch_labels
                    batch_waves = []
                    batch_labels = []
                audio = audio[sample_size:]
                # librosa.output.write_wav('./test.wav', np.array(audio), sample_rate, norm=True)
                # print('done')
                # sleep(1000)
                print("batch_size: {}".format(len(batch_waves)))
            print("load {} files.".format(file_count))
            file_count += 1


def random_pick_to_test_dataset(vocal_directory, non_vocal_directory, test_data_directory):
    files = find_files(vocal_directory)
    files.extend(find_files(non_vocal_directory))
    output_vocal_dir = os.path.join(test_data_directory, 'vocal')
    output_non_vocal_dir = os.path.join(test_data_directory, 'non_vocal')

    if not os.path.exists(output_vocal_dir):
        os.makedirs(output_vocal_dir)
    if not os.path.exists(output_non_vocal_dir):
        os.makedirs(output_non_vocal_dir)
    picked = random.sample(files, len(files) // 10)
    for file_path in picked:
        filename = os.path.basename(file_path)
        if filename[0] == 'n':
            os.rename(file_path, os.path.join(output_non_vocal_dir, filename))
        else:
            os.rename(file_path, os.path.join(output_vocal_dir, filename))


def put_back_test_dataset(test_dir, vocal_dir, non_vocal_dir):
    files = find_files(test_dir)
    for file_path in files:
        filename = os.path.basename(file_path)
        if filename[0] == 'n':
            os.rename(file_path, os.path.join(non_vocal_dir, filename))
        else:
            os.rename(file_path, os.path.join(vocal_dir, filename))


# random_pick_to_test_dataset('./vocal', './non_vocal', './test_data')
# put_back_test_dataset('./test_data', './vocal', './non_vocal')


class AudioReader:
    def __init__(self,
                 coord,
                 queue_size=256,
                 min_after_dequeue=20,
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
            for (vocal_audio, vocal_file), (non_vocal_audio, non_vocal_file) in zip(vocal_iterator, non_vocal_iterator):
                if self.coord.should_stop():
                    stop = True
                    break
                print("vocal: {}".format(vocal_file))
                print("non vocal: {}".format(non_vocal_file))
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
