from tools import AlgorithmicSupervisor, DisturbanceGenerator
from Correct_Trajectories import CorrectTraj
from utils import now_stamp
import torch
import os


class DisturbancedDemonstration:
    def __init__(self, disturbance=None):
        yd_string, time_string = now_stamp()
        dir_ID = yd_string
        self.data_dir = 'Data/Disturbances/tensors/' + dir_ID + '/'
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.env = CorrectTraj(envname="Myenv:slit-v0")
        self.disturbance_generator = disturbance.disturbance_generator

        # interval = 5
        # optimal_traj_R = torch.load("Data/Trajectory/optimal_traj1.pickle")[::interval]
        # optimal_traj_L = torch.load("Data/Trajectory/optimal_traj1.pickle")[::interval]

        optimal_traj_R = torch.load("Data/Trajectory/Hetero_expert_R.pickle")
        optimal_traj_L = torch.load("Data/Trajectory/Hetero_expert_L.pickle")
        self.optimal_traj = [optimal_traj_L, optimal_traj_R]

        self.sup = AlgorithmicSupervisor(
            action_dim=2, optimal_traj=self.optimal_traj)

    def main(self):
        MAX_TRAJ = 1
        i = 0
        while i < MAX_TRAJ:
            i += 1

            S, A, R = self.env.sample(
                1, policy=self.sup, render=True, cov=torch.tensor([0.4, 0.1]))
            injected_d_x = self.env.results['Trajectories']['Fail'][0][2][:, 0]
            injected_d_y = self.env.results['Trajectories']['Fail'][0][2][:, 1]
            # S, A, R = self.env.sample(
            #     1, policy=self.sup, render=True, cov=self.disturbance_generator)
            # injected_d_x = self.env.results['Trajectories']['Success'][0][2][:, 0]
            # injected_d_y = self.env.results['Trajectories']['Success'][0][2][:, 1]
            injected_disturbance_levels = (
                injected_d_x**2 + injected_d_y**2).sqrt()
            _, file_name = now_stamp()
            torch.save(injected_disturbance_levels,
                       'Data/Disturbances/' + file_name + '.pickle')


if __name__ == "__main__":
    policy = torch.load("Data/MHGP-BDI/0/1/learning.pickle")["learner"]
    disturbance_generator = DisturbanceGenerator(policy=policy)
    env = DisturbancedDemonstration(disturbance=disturbance_generator)
    env.main()
