import numpy as np
import torch


class Learner:
    def __init__(self, policy, Learning=True):
        self.Learning_flag = Learning
        self.policy = policy
        pass

    def reset(self):
        pass

    def action_decision(self, state, goal_flag=False):
        if self.policy.__class__.__name__ == 'IOMHGP':
            action, var, _ = self.policy.predict(state)
        else:
            action, var = self.policy.predict(state)

        if self.Learning_flag:
            action = action[int(goal_flag), 0]
        else:
            value, label = var.min(0)
            action = action[label[0], 0].diag()
            var = var[label[0], 0].diag()

        return action, var
