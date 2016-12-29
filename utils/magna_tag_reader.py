import csv
import os
from shutil import copyfile
from subprocess import call
from time import sleep

from definitions import ROOT_DIR
import librosa


class MagnaTagReader:
    def __init__(self):
        self.tag_file = open(os.path.join(ROOT_DIR, './mp3/annotations_final.csv'))
        self.mp3_dir = os.path.join(ROOT_DIR, './mp3')
        self.output_vocal_dir = os.path.join(self.mp3_dir, 'vocal')
        self.output_non_vocal_dir = os.path.join(self.mp3_dir, 'non_vocal')
        if not os.path.exists(self.output_vocal_dir):
            os.makedirs(self.output_vocal_dir)
        if not os.path.exists(self.output_non_vocal_dir):
            os.makedirs(self.output_non_vocal_dir)

    def read_and_copy(self):
        csv_reader = csv.reader(self.tag_file, delimiter=',')
        next(csv_reader)
        for row in csv_reader:
            clip_id = "{}.wav".format(row[0])
            path = row[3]
            print(row)
            # non vocal
            if row[1] == '1':
                os.system("ffmpeg -i {} -acodec pcm_s16le -ac 1 -ar 16000 {}".format(os.path.join(self.mp3_dir, path),
                                                                                os.path.join(self.output_non_vocal_dir,
                                                                                             clip_id)))
            else:
                os.system("ffmpeg -i {} -acodec pcm_s16le -ac 1 -ar 16000 {}".format(os.path.join(self.mp3_dir, path),
                                                                                os.path.join(self.output_vocal_dir,
                                                                                             clip_id)))


MagnaTagReader().read_and_copy()
