from Myenv.slit_env import SlitEnv


class SlitVideoEnv(SlitEnv):
    def __init__(self):
        super().__init__()
        self.ENV_NOISE = 0.0
        # dynamics
        self.total_mass = self.masscart = 1.0
        self.force_scale = 30.0
        # tau -------------------------------------------
        # [collecting] slow : 0.015, fast: 0.02
        # [operating] 0.025
        self.tau = 0.025  # seconds between state updates
        self.kinematics_integrator = "euler"
