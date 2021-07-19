from tools import AlgorithmicSupervisor
from Correct_Trajectories import CorrectTraj
import torch


class Test_Slit:
    def __init__(self):
        self.env = CorrectTraj(envname="Myenv:slit-v0")

        # interval = 5
        # optimal_traj_R = torch.load("Data/Trajectory/optimal_traj1.pickle")[::interval]
        # optimal_traj_L = torch.load("Data/Trajectory/optimal_traj1.pickle")[::interval]

        optimal_traj_R = torch.load("Data/Trajectory/Hetero_expert_R.pickle")
        optimal_traj_L = torch.load("Data/Trajectory/Hetero_expert_L.pickle")
        self.optimal_traj = [optimal_traj_L, optimal_traj_R]
        for i in self.optimal_traj:
            print(i.shape[0])

        self.sup = AlgorithmicSupervisor(action_dim=2, optimal_traj=self.optimal_traj)

    def main(self):
        cov = torch.tensor([0.0, 0.0])
        MAX_TRAJ = 10
        i = 0
        while i < MAX_TRAJ:
            i += 1
            S, A, R = self.env.sample(1, policy=self.sup, render=True, cov=cov)
            print(S.shape[0])


if __name__ == "__main__":
    env = Test_Slit()
    env.main()
