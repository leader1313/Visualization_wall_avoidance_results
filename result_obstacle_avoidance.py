import torch
from VideoMaker_v2 import VideoMaker
import matplotlib.pyplot as plt


class ObstacleAvoidance_VideoMaker(VideoMaker):
    def __init__(self, video_name):
        super().__init__(video_name=video_name)
        self.xlim = 300

    def get_disturbance_figure(self, step):

        # plt.xlim(0, self.dataset.shape[0])
        plt.xlim(0, self.xlim)
        plt.ylim(0, 1.0)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        file_name = self.figure_dir + 'frame' + str(step)
        plt.savefig(file_name)
        plt.clf()

    def get_noisy_figure(self, step):

        # plt.xlim(0, self.dataset.shape[0])
        plt.xlim(0, self.xlim)
        plt.ylim(-1.5, 1.5)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        file_name = self.figure_dir + 'frame' + str(step)
        plt.savefig(file_name)
        plt.clf()

    def get_var_figure(self, step):

        plt.xlim(0, 300)
        plt.ylim(0, 0.2)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        file_name = self.figure_dir + 'frame' + str(step)
        plt.savefig(file_name)
        plt.clf()


if __name__ == '__main__':
    # Import data set
    import numpy as np
    # load_dataset
    # Demonstration ----------------------------------------------------
    # Homo
    MGP_Data = torch.load(
        "/Users/hanbit-o/code/Visualization_wall_avoidance_results/Data/Result/ObstacleAvoidance/MGP-BDI/0/0/learner.pickle")['learner']

    MGP_level = MGP_Data.log_sigma.exp()[0]
    MGP_level_X = MGP_level[0]
    MGP_level_Y = MGP_level[1]
    MGP_level_Z = MGP_level[2]
    MGP_level = (MGP_Data.log_sigma.exp()**2).sum().sqrt()

    MGP_BDI = torch.ones(250, 1) * MGP_level_X
    # Random sampling
    MGP_BDI = torch.normal(
        torch.zeros(MGP_BDI.shape[0]), std=MGP_BDI.squeeze())

    # Hetero
    Hetero_path = "/Users/hanbit-o/code/Visualization_wall_avoidance_results/Data/Result/ObstacleAvoidance/MHGP-BDI/0/1/demo.pickle"
    MHGP_Data = torch.load(Hetero_path)['demo']['Trajectories']['Success'][1]

    S = MHGP_Data[0]  # state
    A = MHGP_Data[1]  # action
    D = MHGP_Data[2]  # disturbances
    # normalization output dims
    MHGP_level = torch.sum(D**2, axis=1).sqrt()
    MHGP_level_X = D[:, 0]
    MHGP_level_Y = D[:, 1]
    MHGP_level_Z = D[:, 2]
    MHGP_BDI = torch.normal(torch.zeros(MHGP_level.shape[0]),
                            MHGP_level_X)

    videoMaker = ObstacleAvoidance_VideoMaker(
        video_name='Obstacle_Avoidance/Homo')
    get_fig = videoMaker.get_noisy_figure
    videoMaker.make_figs(main_dataset=MGP_BDI,
                         main_color='navy', get_fig=get_fig, linewidth=2)
    videoMaker.figure2video()

    videoMaker = ObstacleAvoidance_VideoMaker(
        video_name='Obstacle_Avoidance/Hetero')
    get_fig = videoMaker.get_noisy_figure
    videoMaker.make_figs(main_dataset=MHGP_BDI, sub_dataset=MGP_BDI,
                         main_color='tomato', get_fig=get_fig, sub_alpha=0.3, linewidth=2)
    videoMaker.figure2video()

    # videoMaker = ObstacleAvoidance_VideoMaker(
    #     dataset=disturbances, video_name='Obstacle_Avoidance/Hetero')
    # videoMaker.make_figs(
    #     get_figs=videoMaker.get_hetero_injected_disturbance_figure)
    # videoMaker.figure2video()

    # uncertainty ----------------------------------------------------

    # MGP-BC
    # test_results_path = "/Users/hanbit-o/code/Visualization_wall_avoidance_results/Data/Result/ObstacleAvoidance/MHGP-BC/0/3/20210829_130515/test_results.pickle"
    # trajs = torch.load(test_results_path)['Trajectories']['Fail'][0]

    # S = trajs[0]  # state
    # A = trajs[1]  # action
    # D = trajs[2]  # disturbances

    # learner = torch.load(
    #     "/Users/hanbit-o/code/Visualization_wall_avoidance_results/Data/Result/ObstacleAvoidance/MHGP-BC/0/3/learner.pickle")["learner"]

    # mm, vv, gv = learner.predict(S)
    # value, label = vv.min(axis=0)
    # norm_uncertainty = (value**2).sum(axis=1).sqrt().detach().numpy()
    # videoMaker = ObstacleAvoidance_VideoMaker(
    #     dataset=norm_uncertainty, video_name='Obstacle_Avoidance/covariate_shift')
    # videoMaker.make_figs(
    #     get_figs=videoMaker.get_BC_figure)
    # videoMaker.figure2video()
