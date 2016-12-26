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
import matplotlib.pyplot as plt

class FilterDisplay:
    @classmethod
    def show_weight(cls, weight):
        for i in range(0, weight.shape[2]):
            axes = plt.gca()
            axes.set_ylim([-5, 5])
            plt.plot(weight[:, :, i].flatten())
            plt.show()
            plt.clf()
