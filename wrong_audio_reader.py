import fnmatch
import os
from random import shuffle
from time import sleep
import numpy as np
import librosa


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
        audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
        audio = np.array(audio)
        # audio *= 10
        # audio -=10
        audios.append(audio)
    return audios


class WrongAudioReader:
    def __init__(self,
                 sample_rate=16000,
                 sample_size=1000,
                 vocal_audio_directory='./vocal',
                 non_vocal_audio_directory='./non_vocal'):
        self.sample_rate = sample_rate
        self.sample_size = sample_size
        self.audios = self.read_dir(vocal_audio_directory, self.sample_rate, is_vocal=True)
        self.audios.extend(self.read_dir(non_vocal_audio_directory, self.sample_rate))
        # shuffle(self.audios)
        self.current_audio = None
        self.current_audio_label = None
        self.current_audio_index = 0
        self.current_audio_sample = 0
        self.next_audio_sample()
        self.samples = []
        self.labels = []
        self.end = False
        self.current_batch_index = 0

    def get_all(self, batch_size):
        tuples = []
        while True:
            if self.end:
                break
            samples, labels = self.next_batch(batch_size=batch_size)
            for tuple in zip(samples, labels):
                tuples.append(tuple)
        shuffle(tuples)
        for sample, label in tuples:
            self.samples.append(sample)
            self.labels.append(label)

    def next(self, batch_size):
        if self.current_batch_index + batch_size > len(self.samples):
            self.current_batch_index = 0
        samples_batch = self.samples[self.current_batch_index:self.current_batch_index + batch_size]
        labels_batch = self.labels[self.current_batch_index:self.current_batch_index + batch_size]

        samples_batch = np.array(samples_batch)
        self.current_batch_index += batch_size
        return samples_batch, labels_batch

    def next_audio_sample(self):
        self.current_audio, is_vocal = self.audios[self.current_audio_index]
        self.current_audio_label = [0, 1] if is_vocal == 1 else [1, 0]
        self.current_audio_index += 1
        self.current_audio_index %= len(self.audios)
        if self.current_audio_index == 0:
            self.end = True
        return self.current_audio, self.current_audio_label

    def next_batch(self, batch_size):
        batches_sample = []
        batches_label = []
        while len(batches_sample) < batch_size:
            while self.current_audio_sample + self.sample_size < len(self.current_audio) and \
                            len(batches_sample) < batch_size:
                sample = self.current_audio[self.current_audio_sample:self.current_audio_sample + self.sample_size]
                batches_sample.append(sample)
                batches_label.append(self.current_audio_label)
                self.current_audio_sample += self.sample_size
            if len(batches_sample) < batch_size:
                self.next_audio_sample()
                self.current_audio_sample = 0
                # print("Current audio sample size:{}, now batches size:{}".format(len(self.current_audio),
                #                                                                  len(batches_sample)))

        # print("Current Audio Complete {}/{}".format(self.current_audio_index, len(self.audios)))
        return batches_sample, batches_label

    def read_dir(self, dir, sample_rate, is_vocal=False):
        audio_files = []
        label = 1 if is_vocal else 0
        for audio in load_generic_audio(dir, sample_rate):
            audio_files.append((audio, label))
        return audio_files
