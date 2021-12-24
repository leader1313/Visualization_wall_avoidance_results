import torch

# from GamePad import *
from tools import GamePad


class HumanSupervisor:
    def __init__(self):
        self.game_pad = GamePad(0)
        self.action, _ = self.joyInput()
        pass

    def joyInput(self):
        self.game_pad.Update()
        axes = [i * 50 for i in (self.game_pad.axes_[:2])]
        buttons = self.game_pad.buttons_[:4]
        return axes, buttons

    def reset(self):
        pass

    def action_decision(self, state, goal_flag=None):
        """
        action1: Velocity of vertical Components
        """
        axes, _ = self.joyInput()
        action = torch.tensor([axes[0], -axes[1]]).float()

        return action


class AlgorithmicSupervisor:
    def __init__(self, action_dim, optimal_traj=None):
        self.action_dim = action_dim
        self.optimal_traj = optimal_traj
        pass

    def reset(self):
        self.E = torch.zeros(self.action_dim)
        self.int_E = torch.zeros(self.action_dim)
        self.junction = 0

    def perpendicular_foot(self, C_pose, O_pose, G_pose):
        """Function to output the foot of perpendicular coordinates
        Args
            C_pose : Current pose (x,y)
            O_pose : Object pose (x,y)
            G_pose : Goal pose (x,y)
        """
        a = G_pose - O_pose
        a = a[1] / a[0]
        b = O_pose[1] - a * O_pose[0]
        c = C_pose[0] + a * C_pose[1]

        x = (c - a * b) / (a ** 2 + 1)
        y = a * x + b
        foot = torch.tensor([x, y])

        return foot

    def PID_controller(self, C_pose, G_pose, P=5, D=0.01, I=0.01, old_E=0, old_IE=0):
        """
        C_pose : Current pose (x,y)
        G_pose : Goal pose (x,y)
        DE : Derivative Error
        """
        E = G_pose - C_pose
        DE = E - old_E

        IE = old_IE
        abs_E = E @ E.t()

        # if abs_E > 0.1:
        #     action = (E / abs_E)
        # else:
        action = P * E + D * DE + I * IE
        IE += E

        return action, DE, IE

    def action_decision(self, state, goal_flag=False, goal_state=None):
        """
        action1: Velocity of vertical Components
        action2: Velocity component to the target
        """
        # C_pose = torch.tensor([state[0], state[2]])
        C_pose = state
        # optimal point ----------------------------------
        # G_pose = torch.tensor([8.25, -0.1])
        # trajectory tracking ----------------------------
        G_state = goal_state
        # G_pose = torch.tensor([G_state[0], G_state[2]])
        G_pose = G_state
        action, self.E, self.int_E = self.PID_controller(
            C_pose,
            G_pose,
            P=1,
            D=0.01,
            I=0.0,
            # P=5,
            # D=0.1,
            # I=0.0,
            old_E=self.E,
            old_IE=self.int_E,
        )

        action = torch.clamp(action, min=-1, max=1).float()

        return action


class TrajectoryMaker(AlgorithmicSupervisor):
    def __init__(self, goal_flag, action_dim):
        super().__init__(action_dim)

        if goal_flag == "L":
            self.goal_flag = -1
        else:
            self.goal_flag = 1

    def action_decision(self, state, goal_flag=False, goal_state=None):
        C_pose = state.squeeze()
        # optimal trajectory making --------------------
        flag = self.goal_flag  # left: -1, right: 1
        sub1 = torch.tensor([8.25 * flag, 1.0])
        sub2 = torch.tensor([8.25 * flag, -2.0])
        sub3 = torch.tensor([8.25 * flag, -6.0])
        sub4 = torch.tensor([4.125 * flag, -8.0])
        goal = torch.tensor([0.0, -9.1])
        # goals = [sub1, sub2, sub3, sub4, goal]
        goals = [sub1, sub2, sub3, goal]
        # goals = [sub1, sub2, goal]
        distance = ((C_pose - goals[self.junction]) ** 2).sum()
        # if distance < 0.01:
        if distance < 0.02:
            self.junction += 1
            # self.E = torch.zeros(self.action_dim)
            # self.int_E = torch.zeros(self.action_dim)
        G_pose = goals[self.junction]

        action, self.E, self.int_E = self.PID_controller(
            C_pose,
            G_pose,
            P=1,
            D=0.01,
            I=0.0,
            # P=5, D=0.1, I=0.0,
            old_E=self.E,
            old_IE=self.int_E,
        )
        # Cautious expert
        # action *= 0.5
        # if self.junction == 1 or self.junction == 2:
        #     # action *= 0.3
        #     action = action / abs(action)
        #     action *= 0.2
        # elif self.junction == 3:
        #     action *= 0.5

        # Rough expert
        action *= 0.5
        # if self.junction == 1 or self.junction == 2:
        #     # action *= 0.3
        #     action = action / abs(action)
        #     action *= 0.2
        # elif self.junction == 3:
        #     action *= 0.5

        return action.float()


class ComplexTrajectoryMaker(AlgorithmicSupervisor):
    def __init__(self, goal_flag):
        super().__init__(action_dim=2)
        if goal_flag == 0:
            self.goal_flag = -0.975
        elif goal_flag == 1:
            self.goal_flag = -0.325
        elif goal_flag == 2:
            self.goal_flag = 0.325
        elif goal_flag == 3:
            self.goal_flag = 0.975

    def action_decision(self, state, goal_flag=False, goal_state=None):
        C_pose = state.squeeze()
        # optimal trajectory making --------------------
        flag = self.goal_flag  # h1:-0.975, h2:-0.325, h3:0.325, h4:0.975

        sub1 = torch.tensor([8.25 * flag, 0.5])
        sub2 = torch.tensor([8.25 * flag, -2.0])
        sub3 = torch.tensor([8.25 * flag, -6.0])
        goal = torch.tensor([0.0, -9.1])
        goals = [sub1, sub2, sub3, goal]
        # if abs(self.goal_flag) > 0.5:
        #     sub0 = torch.tensor([8.25 * flag, 9.0])
        #     goals = [sub0, sub1, sub2, sub3, goal]
        # goals = [sub1, sub2, goal]
        distance = ((C_pose - goals[self.junction]) ** 2).sum()
        # if distance < 0.01:
        if distance < 0.02:
            self.junction += 1
        # if distance < 0.5 and self.junction < len(goals) - 1:
        #     self.junction += 1

        G_pose = goals[self.junction]

        action, self.E, self.int_E = self.PID_controller(
            C_pose,
            G_pose,
            P=0.7,  # * abs(flag),
            D=0.01,
            I=0.0,
            # P=5, D=0.1, I=0.0,
            old_E=self.E,
            old_IE=self.int_E,
        )
        # Cautious expert
        # action *= 0.5
        if self.junction == 1 or self.junction == 2:
            # action *= 0.3
            action = action / abs(action)
            action *= 0.2
        # elif self.junction == 3:
        #     action *= 0.5
        action = torch.clamp(action, min=-1, max=1).float()
        return action.float()
