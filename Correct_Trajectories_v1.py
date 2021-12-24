# import numpy as np
import sys
from tools.Controller.Supervisor import TrajectoryMaker
from tools.Utils.repository import Repository
from tools.Utils.Utils import point_approximation
import torch
import gym

sys.path.append("gp_pytorch")


class CorrectTraj:
    def __init__(self, envname):
        self.env = gym.make(envname)
        self.STEP_LIMIT = self.env.STEP_LIMIT
        self.results = {}

    def episode(self, policy=None, render=False, goal_flag=False, cov=0.0):
        policy.reset()
        obs = self.env.reset()
        done = False
        death = False
        step = 0

        state = self.obs_to_state(obs)

        s = []  # state
        a = []  # action
        r = []  # reward
        d = []  # disturbance

        while not done and not death and step < self.STEP_LIMIT:
            step += 1
            if render:
                """render mode
                human: defalt setting
                rgb_array: customized small frame
                """
                self.env.render(mode="human")
                # self.env.render(mode='rgb_array')

            command = torch.empty(self.env.actionsize)
            # action ------------------------------------
            if policy.__class__.__name__ == "AlgorithmicSupervisor":
                optimal_traj = policy.optimal_traj[goal_flag]
                if step < len(optimal_traj):
                    action = policy.action_decision(
                        state, goal_state=optimal_traj[step]
                    )
                else:
                    action = policy.action_decision(
                        state, goal_state=optimal_traj[-1])
            elif policy.__class__.__name__ == "ComplexTrajectoryMaker":
                action = policy.action_decision(state)
            else:
                action = policy.action_decision(
                    state[None, :], goal_flag=goal_flag)

            # noise injection ----------------------------
            try:
                # scalar
                command = torch.normal(mean=action, std=cov)
                d.append(cov)

            except:
                # function
                cov_func = cov
                _, _, cov_v = cov_func(state[None, :])
                cov_s = torch.sqrt(cov_v)
                # print(cov_s)
                command = torch.normal(mean=action, std=cov_s[0])
                # print(cov_s[0])
                d.append(cov_s[0])

            # command = torch.clamp(command, min=-1.0, max=1.0)
            # print(command)

            obs_dash, reward, done, death = self.env.step(command)

            state_dash = self.obs_to_state(obs_dash)

            s.append(state)
            a.append(action)
            r.append(reward)

            state = state_dash
        self.env.close()

        return torch.stack(s), torch.stack(a), torch.stack(r), torch.stack(d)

    def sample(self, Max_traj=2, Max_iter=2, policy=None, render=False, cov=0.0):
        """
        - Demo evaluating mode
            Sample 10 demonstration for evaluating demo successrate
            demo results

        """
        S = []
        A = []
        R = []
        Trajectories = {"Success": [], "Fail": []}
        which_goal = True
        while torch.tensor(R).sum() < Max_traj:
            i = 0
            i_goal = 0
            print(torch.tensor(R).sum())
            while i < Max_iter:
                if policy.__class__.__name__ == "ComplexTrajectoryMaker":
                    # policy.__init__(goal_flag=torch.randint(0, 4, (1,))) # random
                    policy.__init__(goal_flag=i_goal % 4)  # sequential
                i += 1
                s, a, r, d = self.episode(
                    policy=policy, render=render, goal_flag=which_goal, cov=cov
                )
                print("iter:", i, "reward:", sum(r))
                if all(r == 0):
                    print("FAIL-------------------")
                    Trajectories["Fail"].append([s, a, d])
                else:
                    print("SUCCESS-------------------")
                    which_goal = not which_goal
                    i_goal += 1
                    Trajectories["Success"].append([s, a, d])

                R.append(r[-1])
            R = torch.cat(R)
            print("Demo results: {} S | {} F".format(
                R.sum(), Max_iter - R.sum()))
            mm, ss = point_approximation(R * 100)
            print("mean {}% | std {}%".format(mm, ss))

        # if mm >= 50:
        # Append two success trajs
        S = [Trajectories["Success"][:Max_traj][i][0] for i in range(Max_traj)]
        A = [Trajectories["Success"][:Max_traj][i][1] for i in range(Max_traj)]

        self.results["Trajectories"] = Trajectories
        try:
            return torch.cat(S), torch.cat(A), R
        except:
            return S, A, R

    def obs_to_state(self, obs):
        # return torch.tensor(obs).float()
        state = torch.tensor([obs[0], obs[2]]).float()
        return state


if __name__ == "__main__":
    from tools import (
        HumanSupervisor,
        AlgorithmicSupervisor,
        TrajectoryMaker,
        GamePad,
        Repository,
    )

    # params: tau=0.02, Step_Limit = 1000,

    repo = Repository()
    env_name = "wideslit-v0"
    # env = CorrectTraj(envname="Myenv:slit-v0")
    env = CorrectTraj(envname="Myenv:" + env_name)
    policy = TrajectoryMaker(action_dim=2, goal_flag="R")
    # policy = HumanSupervisor()
    cov = torch.tensor([0.0, 0.0])

    MAX_TRAJ = 1
    i = 0
    while i < MAX_TRAJ:
        i += 1
        S, A, R = env.sample(3, policy=policy, render=True, cov=cov)
        repo.save_data(
            S,
            dir_path="Data/Trajectory/",
            file_name="test_traj_" + env_name + str(i),
        )
        print(S.shape[0])
