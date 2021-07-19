from typing_extensions import Annotated
import torch
import os
import matplotlib.pyplot as plt

plt.style.use("ggplot")


class drawFigures:
    """
    draw several type of figures
    """

    def __init__(self):
        pass

    def save_figure(self, dir_fig=str, fig_name=str):
        if not os.path.exists(dir_fig):
            os.makedirs(dir_fig)
        plt.savefig(
            dir_fig + fig_name + ".pdf",
            bbox_inches="tight",
        )

    def print_text_list(self, text_list, x_position):
        text_ind = 0
        axes = plt.gca()
        y_min, y_max = axes.get_ylim()
        for text in text_list:
            text_ind += 1
            plt.text(
                x_position,
                (y_min - 4) - 5 * text_ind,
                str(text),
                fontsize=10,
                style="oblique",
                ha="center",
                va="top",
                wrap=True,
            )

    def get_performance_fig(self, performance={}, dir_fig=str, fig_name=str):
        success_rate = performance["success_rate"]
        iter_length = len(success_rate)
        mm = torch.stack([success_rate[i]["mean"] for i in range(iter_length)])
        ss = torch.stack([success_rate[i]["std"] for i in range(iter_length)])
        params = performance["params"]
        max_iter = params["iters"]
        dataset_dirs = params["repo_dir"]

        iterations = torch.linspace(0, iter_length - 1, iter_length)
        line = plt.plot(iterations, mm)
        std = plt.fill_between(
            iterations, mm - ss, mm + ss, color=line[0].get_color(), alpha=0.1
        )
        xticks = torch.linspace(0, max_iter, steps=max_iter + 1)
        plt.xticks(xticks, fontsize=15)
        plt.yticks([0, 25, 50, 75, 100], fontsize=15)
        annotated_params = {
            "title": params["title"],
            "policy_model": params["policy_model"],
            "lengthscale": params["lengthscale"],
            "Max_learning_iter": params["Max_learning_iter"],
        }
        text_list = [
            str(annotated_params),
            dataset_dirs,
        ]
        self.print_text_list(text_list=text_list, x_position=torch.mean(xticks))
        self.save_figure(dir_fig=dir_fig + "performance/", fig_name=fig_name)
        plt.close()

    def get_demo_rate_fig(self, data={}, dir_fig=str, fig_name=str):
        D = data["demo_success_rate"]
        iter_length = len(D)
        mm = torch.stack([D[i]["mean"] for i in range(iter_length)])
        ss = torch.stack([D[i]["std"] for i in range(iter_length)])
        params = data["params"]
        max_iter = params["Max_learning_iter"]
        dataset_dirs = params["repo_dir"]

        iterations = torch.linspace(0, iter_length - 1, iter_length)
        line = plt.plot(iterations, mm)
        std = plt.fill_between(
            iterations, mm - ss, mm + ss, color=line[0].get_color(), alpha=0.1
        )
        xticks = torch.linspace(0, max_iter, steps=max_iter + 1)
        plt.xticks(xticks, fontsize=15)
        plt.yticks([0, 25, 50, 75, 100], fontsize=15)
        text_list = [str(params), dataset_dirs]
        self.print_text_list(text_list=text_list, x_position=torch.mean(xticks))
        self.save_figure(dir_fig=dir_fig + "demo/", fig_name=fig_name)
        plt.close()

    def get_MVB_SR_fig(self, data={}, dir_fig=str, fig_name=str):
        """
        Describe relationship between marginalized variational bound and success rate.
        X: Success_rate
        Y: MVB
        """
        success_rate = data["Success_rate"]
        VB = data["Marginalized_VB"]
        params = data["params"]
        plt.scatter(VB, success_rate)

        xticks = torch.linspace(min(VB), max(VB), steps=10)
        plt.xticks(xticks, fontsize=15)
        plt.yticks([0, 25, 50, 75, 100], fontsize=15)
        text_list = [str(params), data["kernel_params"]]
        self.print_text_list(text_list=text_list, x_position=torch.mean(xticks))
        self.save_figure(dir_fig=dir_fig + "relearning/", fig_name=fig_name)
        plt.close()

    def get_traj_plt(self, traj, color="gray"):
        # trajectroies
        [state, action, disturbance] = traj
        [X, Y] = state.permute(1, 0)
        plt.plot(X, Y, color=color, alpha=0.1)

    def get_traj_fig(self, dir_fig=str, fig_name=str):
        # back ground image
        im = plt.imread("Data/Figures/slit_env.png")
        plt.imshow(im, extent=[-10, 10, -10, 10])
        xticks = torch.linspace(-10, 10, 21)
        yticks = torch.linspace(-10, 10, 21)
        plt.xticks(xticks)
        plt.yticks(yticks)
        self.save_figure(dir_fig=dir_fig + "Trajectories/", fig_name=fig_name)

    def get_demos_fig(self, demos, dir_fig=str, fig_name=str):
        # back ground image
        im = plt.imread("Data/Figures/slit_env.png")
        plt.imshow(im, extent=[-10, 10, -10, 10])
        xticks = torch.linspace(-10, 10, 21)
        yticks = torch.linspace(-10, 10, 21)
        plt.xticks(xticks)
        plt.yticks(yticks)
        # get demos
        s_trajs = [
            demos[i]["demo"]["Trajectories"]["Success"] for i in range(len(demos))
        ]
        S_T = []
        for t in s_trajs:
            for i in range(len(t)):
                S_T.append(t[i])
        for t in S_T:
            self.get_traj_plt(t)
        params = demos[0]["param"]
        text_list = [str(params)]
        self.print_text_list(text_list=text_list, x_position=torch.mean(xticks))
        self.save_figure(dir_fig=dir_fig + "Trajectories/", fig_name=fig_name)
        # plt.show()
        plt.close()

    def get_demo_fig(self, demo, dir_fig=str, fig_name=str):
        # back ground image
        im = plt.imread("Data/Figures/slit_env.png")
        plt.imshow(im, extent=[-10, 10, -10, 10])
        xticks = torch.linspace(-10, 10, 21)
        yticks = torch.linspace(-10, 10, 21)
        plt.xticks(xticks)
        plt.yticks(yticks)
        # get demo
        s_trajs = demo["Trajectories"]["Success"]
        f_trajs = demo["Trajectories"]["Fail"]
        for t in s_trajs:
            self.get_traj_plt(t)
        for t in f_trajs:
            self.get_traj_plt(t, color="red")

        annotated_param = {
            "success_rate": demo["demo_success_rate"],
            "injection_noise": demo["injection_noise"],
        }
        text_list = [
            str(annotated_param),
            demo["dir"],
        ]
        self.print_text_list(
            text_list=text_list,
            x_position=torch.mean(xticks),
        )
        # No grid line
        plt.grid(b=None)
        # No axis
        plt.axis("off")
        self.save_figure(dir_fig=dir_fig + "Trajectories/", fig_name=fig_name)
        # plt.show()
        plt.close()

    def get_tests_fig(self, tests, dir_fig=str, fig_name=str):
        # back ground image
        im = plt.imread("Data/Figures/slit_env.png")
        plt.imshow(im, extent=[-10, 10, -10, 10])
        xticks = torch.linspace(-10, 10, 21)
        yticks = torch.linspace(-10, 10, 21)
        plt.xticks(xticks)
        plt.yticks(yticks)
        # get demos
        s_trajs = [
            tests[i]["test_results"]["Trajectories"]["Success"]
            for i in range(len(tests))
        ]
        S_T = []
        for t in s_trajs:
            for i in range(len(t)):
                S_T.append(t[i])

        for traj in S_T:
            self.get_traj_plt(traj)
        self.get_traj_fig()
        params = tests[0]["param"]
        text_list = [str(params)]
        self.print_text_list(text_list=text_list, x_position=torch.mean(xticks))
        self.save_figure(dir_fig=dir_fig + "Trajectories/", fig_name=fig_name)
        # plt.show()
        plt.close()


if __name__ == "__main__":
    dF = drawFigures()
    path = "Data/20210528/0/MGP-BC/"
    performance = {}
    dirs = [path + str(i) + "/" for i in range(10)]
    performance["success_rate"] = [
        torch.load(dirs[i] + "test.pickle")["test_results"]["success_rate"]
        for i in range(10)
    ]
    performance["params"] = {"Max_learning_iter": 10}
    dF.get_performance_fig(performance, dir_fig="Data/Figures/", fig_name="test")
