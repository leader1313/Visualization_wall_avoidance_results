from VideoMaker import VideoMaker
from Correct_Trajectories import CorrectTraj
from tools import Learner
import matplotlib.pyplot as plt
import torch
'''
test
learner
data
variance (uncertainty)
'''


class RecordTestLearner:
    def __init__(self, learner=None):
        self.env = CorrectTraj(envname="Myenv:slit-v0")
        self.learner = learner

        policy = torch.load("Data/MHGP-BDI/0/3/learning.pickle")["learner"]
        self.BDI = Learner(policy=policy, Learning=False)
        policy = torch.load("Data/MHGP-BDI/0/2/learning.pickle")["learner"]
        self.BC = Learner(policy=policy, Learning=False)

    def test(self, learner):
        S, A, V, R = self.env.sample(
            1, policy=learner, render=True, cov=torch.tensor([0.0, 0.0])
        )
        self.results = self.env.results
        self.traj = self.results["Trajectories"]
        variances = V.norm(dim=1).detach()
        return variances

    # snapshots from learning process -------

    def get_BC_figure(self, step, dir=None):

        plt.style.use("ggplot")
        var = self.BC_variances

        X = torch.linspace(1, step + 1, steps=step + 1)
        Y = var[:step+1]

        # plt.xlim(0, self.dataset.shape[0])
        plt.xlim(0, 300)
        plt.ylim(0, 3.0)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        # BC
        line = plt.plot(X, Y, color='navy', linewidth=5)

        file_name = dir + 'frame' + str(step)
        plt.savefig(file_name)
        plt.clf()

    def get_BDI_figure(self, step, dir=None):

        plt.style.use("ggplot")
        var = self.BDI_variances

        X = torch.linspace(1, step + 1, steps=step + 1)
        Y = var[:step+1]

        # plt.xlim(0, self.dataset.shape[0])
        plt.xlim(0, 300)
        plt.ylim(0, 3.0)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        # BC
        fail_max_step = self.BC_variances.shape[0]
        xx = torch.linspace(1, fail_max_step, steps=fail_max_step)
        yy = self.BC_variances
        baseline = plt.plot(xx, yy, color='navy', linewidth=5)

        # BDI
        line = plt.plot(X, Y, color='tomato', linewidth=5)

        file_name = dir + 'frame' + str(step)
        plt.savefig(file_name)
        plt.clf()

    def make_graph_videos(self, get_figs):
        self.video_maker.make_figs(get_figs=get_figs)
        self.video_maker.figure2video()

    def reset_env(self):
        self.video_maker = VideoMaker(
            dataset=None, video_name='Variance')
        self.env = CorrectTraj(envname="Myenv:slit-v0",
                               monitoring_dir=self.video_maker.file_name)

    def main(self):
        for _ in range(10):
            self.reset_env()
            self.BC_variances = self.test(learner=self.BC)
            self.video_maker.dataset = self.BC_variances
            self.make_graph_videos(get_figs=self.get_BC_figure)
            self.reset_env()
            self.BDI_variances = self.test(learner=self.BDI)
            self.video_maker.dataset = self.BDI_variances
            self.make_graph_videos(get_figs=self.get_BDI_figure)


if __name__ == "__main__":
    policy = torch.load("Data/MHGP-BDI/0/2/learning.pickle")["learner"]
    learner = Learner(policy=policy, Learning=False)
    test = RecordTestLearner(learner=learner)
    test.main()
    # test.make_graph_videos()
