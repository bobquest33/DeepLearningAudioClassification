from unittest import TestCase

import librosa
import matplotlib.pyplot as plt
from audio_reader import AudioReader
import numpy as np
import pyglet


class TestAudioReader(TestCase):
    def setUp(self):
        self.audio_reader = AudioReader(sample_size=32000)
        self.audio_reader.get_all_batches()
        self.audio_reader.tied_batches()

    def test_next_batch(self):
        i = 0
        while True:
            samples, labels, _ = self.audio_reader.next_batch(batch_size=20)
            if self.audio_reader.batch_is_rolling_back():
                break
            for sample, label in zip(samples, labels):
                # vocal
                if label[0] == 0:
                    librosa.output.write_wav('./test_vocal/{}.wav'.format(i), sample, self.audio_reader.sample_rate)
                else:
                    librosa.output.write_wav('./test_non_vocal/{}.wav'.format(i), sample, self.audio_reader.sample_rate)
                i += 1
