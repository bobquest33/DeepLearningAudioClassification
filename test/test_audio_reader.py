from unittest import TestCase

from audio_reader import AudioReader
import numpy as np

class TestAudioReader(TestCase):
    def setUp(self):
        self.audio_reader = AudioReader(sample_size=20)
        self.sample_buckets = {}
        self.label_buckets = {}

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
        audio, label = self.audio_reader.next_batch(batch_size=32)
        print(audio.shape, label)
