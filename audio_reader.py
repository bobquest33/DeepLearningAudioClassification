import fnmatch
import os
from random import shuffle, random
from time import sleep
import numpy as np
import librosa
import random

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
    audios = []
    for filename in files:
        audio, sr = librosa.load(filename, sr=sample_rate, mono=True)
        audios.append(audio)
    return audios


class AudioReader:
    def __init__(self,
                 sample_rate=16000,
                 sample_size=20,
                 vocal_audio_directory='./vocal',
                 non_vocal_audio_directory='./non_vocal'):
        self.sample_rate = sample_rate
        self.sample_size = sample_size
        self.audios = self.read_dir(vocal_audio_directory, self.sample_rate, is_vocal=True)
        self.audios.extend(self.read_dir(non_vocal_audio_directory, self.sample_rate))
        shuffle(self.audios)
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
