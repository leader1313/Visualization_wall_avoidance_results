from tools import Learner
from Correct_Trajectories import CorrectTraj
import torch


class Test:
    def __init__(self, learner=None, env=CorrectTraj(envname="Myenv:slit-v0")):
        self.learner = learner
        self.env = env
        pass

    def testLearner(self):
        S, A, R = self.env.sample(
            10, policy=self.learner, render=True, cov=torch.tensor([0.0, 0.0])
        )


if __name__ == "__main__":
    policy = torch.load("Data/MHGP-BDI/0/5/learning.pickle")["learner"]
    learner = Learner(policy=policy, Learning=False)
    test = Test(learner=learner)
    test.testLearner()
