import csv
import os

from definitions import ROOT_DIR


class MagnaTagReader:
    def __init__(self):
        self.tag_file = open(os.path.join(ROOT_DIR, 'annotations_final.csv'))
        self.mp3_dir = './mp3'

    def read(self):
        for row in csv.reader(self.tag_file, delimiter=','):
            if row[1] == '1':


MagnaTagReader().read()
