import numpy as np
import torch


class DisturbanceGenerator:
    def __init__(self, policy):
        self.policy = policy

    def disturbance_generator(self, state):
        if self.policy.__class__.__name__ == 'IOMHGP':
            return self.policy.predict(state)
        else:
            disturbance = torch.exp(self.policy.log_sigma[-1, :])
            return torch.sqrt(disturbance)
