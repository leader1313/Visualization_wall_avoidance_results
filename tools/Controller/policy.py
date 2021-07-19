

class Policy():
    def __init__(self, function):
        self.function = function

    def action_decision(self, state):
        """
            To be implemented by supervisor, learner
        """
        raise NotImplementedError


class PIDPolicy():
    def __init__(self):
        self.old_Error = 0.0
        self.old_IE = 0.0
        pass

    def reset_Error(self):
        self.old_Error = 0.0
        self.old_IE = 0.0

    def perpendicular_foot(self, E_pose, O_pose, G_pose):
        '''Function to output the foot of perpendicular coordinates
        Args
            E_pose : End effectro pose (x,y)
            O_pose : Object pose (x,y)
            G_pose : Goal pose (x,y)
        '''
        a = G_pose - O_pose
        a = a[1] / a[0]
        b = O_pose[1] - a * O_pose[0]
        c = E_pose[0] + a * E_pose[1]

        x = (c - a * b) / (a**2 + 1)
        y = a * x + b

        return torch.tensor([x, y])

    def PID_controller(self, E_pose, G_pose, P=5, D=0.01, I=0.01, old_Error=0, old_IE=0):
        '''
        E_pose : End effectro pose (x,y)
        G_pose : Goal pose (x,y)
        DE : Derivative Error
        '''
        Error = G_pose - E_pose
        DE = Error - old_Error

        IE = old_IE
        abs_E = Error @ Error.t()

        if abs_E > 0.1:
            action = (Error / abs_E)
        else:
            action = P * Error + D * DE + I * IE
            IE += Error

        return action.numpy(), DE, IE

    def action_decision(self, state, goal_flag=False):
        state = state[0]
        O_pose = state[0:2]
        if goal_flag:
            G_pose = state[3:5]
        else:
            G_pose = state[6:8]

        E_pose = state[9:11]

        H_pose = self.perpendicular_foot(E_pose, O_pose, G_pose)
        action, self.old_Error, self.old_IE =\
            self.PID_controller(E_pose, H_pose, P=10, I=0.1,
                                old_Error=self.old_Error, old_IE=self.old_IE)

        action *= 3
        return action
