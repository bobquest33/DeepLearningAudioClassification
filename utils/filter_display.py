# import numpy as np
# import matplotlib.pyplot as plt
#
# plt.axis([0, 10, 0, 1])
# plt.ion()
#
# for i in range(10):
#     y = np.random.random()
#     plt.scatter(i, y)
#     plt.pause(0.05)
#
# while True:
#     plt.pause(0.05)
import threading
from time import sleep

import matplotlib.pyplot as plt


class FilterDisplay:
    def __init__(self):
        self.thread = None
        self.weight = None

    def show_weight(self, weight):
        self.weight = weight
        plt.clf()
        axes = plt.gca()
        axes.set_ylim([-5, 5])
        plt.plot(self.weight[:, :, 4].flatten())
        if self.thread is None:
            self.thread = threading.Thread(target=self._show_weight)
            self.thread.start()

    def _show_weight(self):
        print('show')
        plt.ion()
        while True:
            plt.pause(0.05)


# filter = FilterDisplay()
# filter.show_weight([1, 2, 3])
# sleep(2)
# filter.show_weight([1, 2, 5])
# sleep(20000)

