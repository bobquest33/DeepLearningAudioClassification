from unittest import TestCase

import librosa
import matplotlib.pyplot as plt
from audio_reader import AudioReader
import numpy as np


class TestAudioReader(TestCase):
    def setUp(self):
        self.audio_reader = AudioReader(sample_size=20)
        self.validate_audio_reader = AudioReader(sample_size=20,
                                            vocal_audio_directory='./validate_vocal',
                                            non_vocal_audio_directory='./validate_non_vocal')

        self.sample_buckets = {}
        self.label_buckets = {}
        self.validate_audio_reader.get_all_batches()
        self.validate_audio_reader.shuffle_batches()

    def test_get_all_batches(self):
        self.audio_reader.get_all_batches()
        for sample, label, index in self.audio_reader.tuples:
            self.sample_buckets[index] = sample
            self.label_buckets[index] = label

        self.audio_reader.shuffle_batches()
        for sample, index in zip(self.audio_reader.samples, self.audio_reader.numbers):
            np.testing.assert_array_equal(self.sample_buckets[index], sample)

        for label, index in zip(self.audio_reader.labels, self.audio_reader.numbers):
            np.testing.assert_array_equal(self.label_buckets[index], label)

    def test_next_batch(self):
        self.audio_reader = AudioReader(sample_size=20)
        self.audio_reader.get_all_batches()
        for sample, label, index in self.audio_reader.tuples:
            self.sample_buckets[index] = sample
            self.label_buckets[index] = label

        self.audio_reader.shuffle_batches()
        audios, labels, numbers = self.audio_reader.next_batch(batch_size=32)
        for audio, label, number in zip(audios, labels, numbers):
            np.testing.assert_array_equal(self.label_buckets[number], label)

        for sample in self.validate_audio_reader.next_batch(batch_size=32)[0]:
            sample = np.reshape(sample, [20, 20])
            sample = np.transpose(sample)
            librosa.display.specshow(sample, x_axis='time')
            plt.colorbar()
            plt.title('MFCC')
            plt.tight_layout()
            plt.show()
