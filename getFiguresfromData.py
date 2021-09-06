from VideoMaker import VideoMaker
import matplotlib.pyplot as plt
from utils import now_stamp
import os
import numpy as np


class VisualizeDataset:
    def __init__(self, data_name):
        self.data_name = data_name
        self.video_maker = VideoMaker(dataset=None, video_name=data_name)

    def new_dir_ID(self):
        yd_string, time_string = now_stamp()
        dir_ID = yd_string + time_string
        self.data_dir = 'Data/' + self.data_name + '/Figures/' + dir_ID + '/'
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def get_disturbance_figure(self, data, color):
        step = data.shape[0]

        # normalization output dims
        norm_level = np.sqrt(np.sum(data.numpy()**2, axis=1))
        plt.style.use("ggplot")

        X = torch.linspace(1, step + 1, steps=step)
        Y = norm_level

        max_steps = 300
        plt.xlim(0, max_steps)
        plt.ylim(0, 1.0)
        plt.xticks(fontsize=0)
        plt.yticks(fontsize=0)

        line = plt.plot(X, Y, color=color, linewidth=2)

        file_name = 'frame' + str(step)
        save_path = self.data_dir + file_name + '.png'
        plt.savefig(save_path)
        plt.clf()

    def MGP_init_fig(self, step, disturbance_level):
        Y = torch.ones(step) * disturbance_level
        X = torch.linspace(1, step + 1, steps=step)
        line = plt.plot(X, Y, color="navy", linewidth=2)

    def disturbance_figs(self, data, color):
        self.seq_making_figs(data, self.get_disturbance_figure, color=color)

    def seq_making_figs(self, data, fig_maker, color):
        step = data.shape[0]

        for t in range(step):
            snap_data = data[:t + 1]
            fig_maker(snap_data, color)


if __name__ == "__main__":
    import torch
    # define graph writer
    VD = VisualizeDataset(data_name="Disturbance_comparison")
    # load_dataset
    trajs = torch.load("demo.pickle")['demo']['Trajectories']['Success']

    # make dummy MGP data
    MGP_BDI = torch.load("learner.pickle")['learner']
    MGP_disturbance = (MGP_BDI.log_sigma.exp()**2).sum().sqrt()
    MGP_D = torch.ones(150, 1) * MGP_disturbance

    n_trajs = len(trajs)
    for i in range(n_trajs):
        S = trajs[i][0]  # state
        A = trajs[i][1]  # action
        D = trajs[i][2]  # disturbances
        VD.new_dir_ID()
        VD.disturbance_figs(MGP_D, color="navy")
        VD.disturbance_figs(D, color="tomato")

    # put data per step into a make graph function
