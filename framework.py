from tools import (
    AlgorithmicSupervisor,
    GaussianKernel,
    IOMGP,
    HumanSupervisor,
    # ComplexTrajectoryMaker,
    IOMHGP,
    Learner,
    Repository,
    drawFigures,
)
from Correct_Trajectories_v1 import CorrectTraj
from Test_Learner import Test
import torch
import numpy as np


TRIALS = 1
"""
TO DO
    - make figures with parameter labeling
    - framework 
        - demonstration
        - learning
        - test
        - make figures
    - Stratege: check figures and find model from Data
"""


class Framework(object):
    """parameters
    args['envname']
    args['iters']
    args['update']
    args['Max_learning_iter']
    args['Max_mixture']
    args['n_target']
    args['interval_sampling']
    args['random_sampling']
    args['policy_model']
    args['supervisor']
    """

    def __init__(self, params):
        self.params = params
        return

    def reset_learner(self, params):
        """
        Initializes new neural network and learner wrapper
        Initializes new Gaussian Process
        params: learning parameters
        """
        M = params["Max_mixture"]
        lengthscale = params["lengthscale"]
        if self.params["random_sampling"] == 0:
            i = self.params["interval_sampling"]
            S, A = self.S[::i], self.A[::i]
        else:
            i = self.params["random_sampling"]
            ind = torch.randperm(self.S.shape[0])[:i]
            S, A = self.S[ind], self.A[ind]

        if self.params["policy_model"] == "IOMGP":
            kern = GaussianKernel()
            learner = IOMGP(
                S, A, kern, Max_mixture=M, K=self.K, lengthscale=lengthscale
            )
        elif self.params["policy_model"] == "IOMHGP":
            fkern = GaussianKernel()
            gkern = GaussianKernel()
            learner = IOMHGP(S, A, fkern, gkern, M=M, lengthscale=lengthscale)
        return learner

    def prologue(self, params):
        """
        Preprocess hyperparameters and initialize learner and supervisor
        """
        self.env = CorrectTraj(envname=params["envname"])
        self.test = Test(envname=params["envname"])

        iters = self.params["iters"]
        title = self.params["title"]
        self.params["repo_dir"] = (
            self.repo.dir_path + title + "/" + str(self.current_trial)
        )
        self.dirs = [
            self.repo.dir_path
            + title
            + "/"
            + str(self.current_trial)
            + "/"
            + str(i)
            + "/"
            for i in range(iters)
        ]
        self.drawFigures = drawFigures()

        K = S = A = torch.tensor([])
        self.S, self.A, self.K = S, A, K

        if self.params["supervisor"] == "algorithmic":
            if self.params["envname"] == "Myenv:wideslit-v0":
                # optimal_traj_R = torch.load(
                #     "Data/Trajectory/Hetero_expert_wideslit-v0_R.pickle"
                # )
                # optimal_traj_L = torch.load(
                #     "Data/Trajectory/Hetero_expert_wideslit-v0_L.pickle"
                # )
                optimal_traj_R = torch.load(
                    "Data/Trajectory/Hetero_expert_R.pickle")
                optimal_traj_L = torch.load(
                    "Data/Trajectory/Hetero_expert_L.pickle")
            else:
                optimal_traj_R = torch.load(
                    "Data/Trajectory/Hetero_expert_R.pickle")
                # optimal_traj_L = torch.load("Data/Trajectory/Hetero_expert_L.pickle")
                optimal_traj_L = torch.load(
                    "Data/Trajectory/Hetero_expert_R.pickle")

            if self.params["exp"] == "dummy_expert":
                optimal_traj_L = optimal_traj_R = torch.load(
                    "Data/Trajectory/20211110/opt_traj_slit-dummy-v0_L_1636521513.pickle"
                )
            if self.params["exp"] == "cautious_expert":
                optimal_traj_L = optimal_traj_R = torch.load(
                    "Data/Trajectory/cautiouness_exp/cautious_expert.pickle"
                )
            if self.params["exp"] == "over_cautious_expert":
                optimal_traj_L = optimal_traj_R = torch.load(
                    "Data/Trajectory/cautiouness_exp/over-cautious_expert.pickle"
                )
            else:
                pass

            self.optimal_traj = [optimal_traj_R, optimal_traj_L]
            sup = AlgorithmicSupervisor(
                action_dim=2,
                optimal_traj=self.optimal_traj,
            )
        elif self.params["supervisor"] == "complex":
            sup = ComplexTrajectoryMaker(goal_flag=1)
        elif self.params["supervisor"] == "human":
            sup = HumanSupervisor()

        self.sup = sup

        return self.params

    def run_iters(self):
        """
        To be implemented by learning methods (e.g. behavior cloning, dart, dagger...)
        """
        raise NotImplementedError

    def run_tests(self):
        for it in range(self.params["iters"]):
            try:
                policy = self.repo.load_data(self.dirs[it] + "learning.pickle")[
                    "learner"
                ]
                learner = Learner(policy=policy, Learning=False)

                self.test_results = self.test.sample(
                    total_test_iter=100, policy=learner, render=False
                )

                self.save_datasets("test", it)

                performance = {}
                performance["success_rate"] = [
                    torch.load(self.dirs[i] + "test.pickle")["test_results"][
                        "success_rate"
                    ]
                    for i in range(it + 1)
                ]
                performance["params"] = self.params
                self.drawFigures.get_performance_fig(
                    performance,
                    dir_fig=self.repo.dir_figures,
                    fig_name=self.repo.dt_string(),
                )
            except:
                pass

    def drawDemos(self):
        demos = [
            torch.load(self.dirs[i] + "demo.pickle")
            for i in range(self.params["end_iters"] + 1)
        ]
        self.drawFigures.get_demos_fig(
            demos=demos, dir_fig=self.repo.dir_figures, fig_name=self.repo.dt_string()
        )

    def run_trial(self):
        """
        Run a trial by first preprocessing the parameters and initializing
        the supervisor and learner. Then run each iterations (not implemented here)
        """
        self.prologue(params=self.params)
        results = self.run_iters()
        self.drawDemos()
        self.run_tests()

        return results

    def run_trials(self, TRIALS):
        """
        Runs and saves all trials. Generates directories under 'results/experts/'
        where sub-directory names are based on the parameters. Data is saved after
        every trial, so it is safe to interrupt program.
        """
        self.repo = Repository()
        for t in range(TRIALS):
            print("\n\nTrial: " + str(t))
            self.current_trial = t
            self.run_trial()
        return

    def save_datasets(self, data_name, i):
        dataset = {
            "params": self.params,
        }
        if data_name == "demo":
            dataset["demo"] = self.env.results
        elif data_name == "learning":
            dataset["learner"] = self.lnr
        elif data_name == "test":
            dataset["test_results"] = self.test_results

        self.repo.save_data(
            dataset, dir_path=self.dirs[i], file_name=data_name)

        string = "+" * 14 + str(i) + " th iter " + \
            data_name + " saved " + "+" * 14
        string_len = len(string)

        print("=" * string_len)
        print(string)
        print("=" * string_len)


if __name__ == "__main__":
    args = {}
    args["envname"] = "Myenv:slit-v0"
    args["iters"] = 10
    args["update"] = 1
    args["Max_learning_iter"] = 1
    args["Max_mixture"] = 5
    args["n_target"] = 2
    args["interval_sampling"] = 1
    args["random_sampling"] = 0
    args["policy_model"] = "IOMHGP"  # 'IOMHGP'
    args["supervisor"] = "algorithmic"  # 'human'
    args["title"] = "BDI"
    framework = Framework(args)
    framework.run_trials(1)
