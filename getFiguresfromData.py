import matplotlib.pyplot as plt
from utils import now_stamp
import os


class GetFiguresfromData:
    def __init__(self, data):
        yd_string, time_string = now_stamp()
        dir_ID = yd_string
        self.data_dir = 'Data/Disturbances/Figures' + dir_ID + '/'
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.data = data

    def disturbance_figure(self, injected_disturbances):
        step = injected_disturbances.shape[0]
        injected_d_x = injected_disturbances[:, 0]
        injected_d_y = injected_disturbances[:, 1]
        injected_disturbance_levels = (
            injected_d_x**2 + injected_d_y**2).sqrt()

        plt.style.use("ggplot")

        X = torch.linspace(1, step + 1, steps=step + 1)
        Y = injected_disturbance_levels

        max_steps = 220
        plt.xlim(0, max(max_steps))
        plt.ylim(0, 1.0)
        plt.xticks(fontsize=0)
        plt.yticks(fontsize=0)

        line = plt.plot(X, Y, )

        file_name = 'frame' + str(step)
        plt.savefig(file_name)
        plt.clf()
