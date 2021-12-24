from tools import (
    AlgorithmicSupervisor,
    GaussianKernel,
    IOMGP,
    Repository,
    HumanSupervisor,
    IOMHGP,
    Learner,
)
import torch
import gym
import time


class Test:
    def __init__(self, envname):
        self.env = gym.make(envname)
        self.STEP_LIMIT = self.env.STEP_LIMIT
        pass

    def episode(self, policy=None, render=False):
        obs = self.env.reset(random_init=True)
        done = False
        death = False
        step = 0
        state = self.obs_to_state(obs)

        s = []
        a = []
        r = []

        while not done and not death and step < self.STEP_LIMIT:
            step += 1
            tic = time.time()
            if render:
                """render mode
                human: defalt setting
                rgb_array: customized small frame
                """
                self.env.render(mode="human")
                # self.env.render(mode='rgb_array')
            toc1 = time.time()

            command = torch.empty(self.env.actionsize)
            # action ------------------------------------
            action = policy.action_decision(state[None, :])
            command = torch.clamp(action, min=-1.0, max=1.0)
            toc2 = time.time()
            obs_dash, reward, done, death = self.env.step(command)
            state_dash = self.obs_to_state(obs_dash)

            toc3 = time.time()

            s.append(state)
            a.append(action)
            r.append(reward)

            state = state_dash
            times = {
                "rendering_time": toc1 - tic,
                "prediction_time": toc2 - toc1,
                "step_time": toc3 - toc2,
            }
            # print("redering time")
            # for key in times:
            #     print(key, " ", times[key])
            #     pass

        self.env.close()
        step_death = step >= self.STEP_LIMIT
        if step_death:
            death = step_death

        return torch.stack(s), torch.stack(a), torch.stack(r), done, death, step_death

    def sample(self, total_test_iter=2, policy=None, render=False):
        Trajectories = {"Success": [], "Fail": []}
        R = []

        i, d, s_d = 0, 0, 0
        N_fail = 0
        N_sim_death, N_step_death = 0, 0
        while i < total_test_iter:
            i += 1
            s, a, r, done, d, s_d = self.episode(policy=policy, render=render)
            if all(r == 0):
                # print("FAIL-------------------")
                N_step_death += s_d
                N_sim_death += d
                N_fail = N_step_death + N_sim_death
                Trajectories["Fail"].append([s, a])
            else:
                # print("SUCCESS-------------------")
                Trajectories["Success"].append([s, a])
            R.append(done)
            success_rate = torch.tensor(R).float() * 100
            did = int(i / (total_test_iter - 1) * 10)
            bar = u"\u2588" * did + " " * (10 - did)
            print("\r [{}] Success rate: {} %".format(
                bar, success_rate.mean()), end="")

        success_rate_mean = success_rate.mean()
        interbal = torch.tensor(success_rate.shape[0]).sqrt().int()
        success_means = torch.stack(
            [success_rate[i: i + interbal].mean() for i in range(interbal)]
        )
        success_rate_std = success_means.std()

        step_death_rate = (N_step_death / total_test_iter) * 100
        sim_death_rate = (N_sim_death / total_test_iter) * 100
        fail_rate = ((N_fail) / total_test_iter) * 100
        total_results = {
            "success_rate": {"mean": success_rate_mean, "std": success_rate_std},
            "step_death_rate": step_death_rate,
            "sim_death_rate": sim_death_rate,
            "fail_rate": fail_rate,
            "Trajectories": Trajectories,
        }

        return total_results

    def obs_to_state(self, obs):
        # return torch.tensor(obs).float()
        state = torch.tensor([obs[0], obs[2]])
        return state


if __name__ == "__main__":
    policy = torch.load("Data/20210528/0/MGP-BC/7/learning.pickle")["learner"]
    learner = Learner(policy=policy, Learning=False)
    test = Test(envname="Myenv:slit-v0")
    results = test.sample(policy=learner, render=True)
    print(results)
