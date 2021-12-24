from tools import AlgorithmicSupervisor, Repository
from tools.Utils.Utils import point_approximation
import framework
import argparse
import torch

"""
python3 BC.py --envname Myenv:FetchPushMultiObjects-v1 --iters 2 --update 1
python3 BC.py --envname Myenv:slit-v0 --iters 2 --update 1
python3 BC.py --policy_model IOMGP --update 1 --interval_sampling 1
"""


def main():
    ap = argparse.ArgumentParser()
    # title of experiments

    ap.add_argument("--title", required=True)

    # ap.add_argument("--policy_model", required=True)

    # ap.add_argument('--envname', required=True)

    # ap.add_argument("--iters", required=True, type=int)
    ap.add_argument("--lengthscale", required=True, type=float)

    # ap.add_argument("--update", required=True, type=int)
    # ap.add_argument("--interval_sampling", required=True, type=int)

    # ap.add_argument('--random_sampling', required=True, type=int)
    # ap.add_argument("--Max_mixture", required=True, type=int)
    # ap.add_argument("--Max_learning_iter", required=True, type=int)

    args = vars(ap.parse_args())
    # args["title"] = "BC_complex"

    args["Max_mixture"] = 10
    args["policy_model"] = "IOMHGP"
    args["iters"] = 6
    # args["lengthscale"] = 4.0

    args["interval_sampling"] = 2
    args["update"] = 2

    # args["envname"] = "Myenv:slit-v0"
    # args["envname"] = "Myenv:wideslit-v0"
    args["envname"] = "Myenv:slit_complex-v0"
    args["Max_learning_iter"] = 10
    args["random_sampling"] = 0
    args["supervisor"] = "complex"  # "algorithmic"  # 'human'
    args["exp"] = "cautious_expert"

    TRIALS = framework.TRIALS

    test = BC(args)
    test.run_trials(TRIALS)


class BC(framework.Framework):
    def run_iters(self):
        n_action = 2
        cov = torch.zeros(n_action)
        demo = {"success_rate": {"mean": [], "std": []}}
        demo["params"] = self.params

        for it in range(self.params["iters"]):
            print("\tIteration: " + str(it))

            s, a, R = self.env.sample(
                Max_traj=self.params["update"],
                Max_iter=10,
                policy=self.sup,
                render=False,
                cov=cov,
            )
            self.save_datasets("demo", it)
            mm, ss = point_approximation(R * 100)

            demo["success_rate"]["mean"].append(mm)
            demo["success_rate"]["std"].append(ss)

            # self.drawFigures.get_demo_rate_fig(
            #     demo,
            #     dir_fig=self.repo.dir_figures,
            #     fig_name=self.repo.dt_string(),
            # )
            self.params["end_iters"] = it

            self.S = torch.cat([self.S, s])
            self.A = torch.cat([self.A, a])
            self.lnr = self.reset_learner(self.params)
            # if (it + 1) % self.params["update"] == 0:
            self.lnr.learning(self.params["Max_learning_iter"])
            self.save_datasets("learning", it)
        return self.lnr


if __name__ == "__main__":
    main()
