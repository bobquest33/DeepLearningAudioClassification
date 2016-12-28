from unittest import TestCase

import librosa
import matplotlib.pyplot as plt
from audio_reader import AudioReader
from wrong_audio_reader import WrongAudioReader
import numpy as np
import tensorflow as tf


class TestAudioThread(tf.test.TestCase):
    def setUp(self):
        self.coord = tf.train.Coordinator()
        self.reader = AudioReader(self.coord)

    def test_thread(self):
        with self.test_session() as sess:
            tf.train.start_queue_runners(sess=sess, coord=self.coord)
            self.reader.start_thread(sess)
            batch = self.reader.dequeue(3)
            audios, label = sess.run(batch)
            label = np.reshape(label, [3, 2])
            audios = np.reshape(audios, [3, 32000])
            print(label)
            index = 0
            for audio in audios:
                librosa.output.write_wav('piece{}.wav'.format(index), audio, 16000)
                index += 1
                # print(sess.run(batch))
            # print(sess.run(y))
            self.coord.request_stop()


class TestAudioReader(TestCase):
    def setUp(self):
        self.audio_reader = AudioReader(sample_size=32000)
        self.audio_reader.get_all_batches()
        self.audio_reader.tied_batches()

    def test_next_batch(self):
        for times in range(1001):
            i = 0
            samples, labels, _ = self.audio_reader.next_batch(batch_size=20)
            for sample, label in zip(samples, labels):
                # vocal
                if times >= 1000:
                    if label[0] == 0:
                        librosa.output.write_wav('./test2_vocal/{}.wav'.format(i), sample,
                                                 self.audio_reader.sample_rate)
                    else:
                        librosa.output.write_wav('./test2_non_vocal/{}.wav'.format(i), sample,
                                                 self.audio_reader.sample_rate)
                    i += 1


class TestWrongAudioReader(TestCase):
    def setUp(self):
        self.audio_reader = WrongAudioReader(sample_size=32000)

    def test_next_batch(self):
        self.audio_reader.get_all(20)
        for times in range(1001):
            i = 0
            samples, labels = self.audio_reader.next(batch_size=20)
            for sample, label in zip(samples, labels):
                if times >= 1000:
                    if label[0] == 0:
                        librosa.output.write_wav('./test_vocal/{}.wav'.format(i), sample, self.audio_reader.sample_rate)
                    else:
                        librosa.output.write_wav('./test_non_vocal/{}.wav'.format(i), sample,
                                                 self.audio_reader.sample_rate)
                    i += 1
