import torch


class ComparisonModels():
    def __init__(self, dir_path, name, color):
        self.dir_path = dir_path
        self.color = color
        self.name = name
        self.demo_success_rate = {}

    def test_results(self, ID):
        dataset = torch.load(
            self.dir_path + '/' + ID + '/_random/test_results.pickle')
        self.success_rate = dataset['success_rate']
        results = torch.stack(
            dataset['test_results']).float() * 100
        interbal = 5
        self.success_rate['std'] = torch.stack(
            [results[i * interbal:i * interbal + interbal].mean() for i in range(int(results.shape[0] / interbal))]).std(unbiased=False)
        print(dataset.keys())

    def demo_results(self, ID):
        dataset = torch.load(self.dir_path + '/' + ID +
                             '/test_demo_results.pickle')
        self.success_rate = dataset['success_rate']
        results = torch.stack(dataset['test_results']).float() * 100
        # results = torch.stack(dataset['test_results']).repeat(10).float() * 100
        interbal = 5
        self.success_rate['std'] = torch.stack(
            [results[i * interbal:i * interbal + interbal].mean() for i in range(int(results.shape[0] / interbal))]).std()
        print(dataset.keys())

    def demo_results_v2(self, ID):
        dataset = torch.load(self.dir_path + '/' + ID +
                             '/demo.pickle')['demo']
        n_success_trajs = len(dataset['Trajectories']['Success'])
        n_fail_trajs = len(dataset['Trajectories']['Fail'])
        print("SUCCESS : ", n_success_trajs)
        print("FAIL : ", n_fail_trajs)
        N = n_success_trajs + n_fail_trajs
        self.demo_success_rate["mean"] = n_success_trajs / N * 100


class MHGP_BDI(ComparisonModels):
    def __init__(self, dir_path):
        name = "MHGP-BDI"
        color = "tomato"
        super().__init__(dir_path, name, color)


class UHGP_BDI(ComparisonModels):
    def __init__(self, dir_path):
        name = "UHGP-BDI"
        color = "steelblue"
        super().__init__(dir_path, name, color)


class MGP_BDI(ComparisonModels):
    def __init__(self, dir_path):
        name = "MGP-BDI"
        color = "sienna"
        super().__init__(dir_path, name, color)


class UGP_BDI(ComparisonModels):
    def __init__(self, dir_path):
        name = "UGP-BDI"
        color = "gold"
        super().__init__(dir_path, name, color)


class MHGP_BC(ComparisonModels):
    def __init__(self, dir_path):
        name = "MGP-BC"
        color = "navy"
        super().__init__(dir_path, name, color)


class UHGP_BC(ComparisonModels):
    def __init__(self, dir_path):
        name = "UGP-BDI"
        color = "grey"
        super().__init__(dir_path, name, color)


if __name__ == "__main__":
    # mhgp_bdi = MHGP_BDI(
    #     dir_path="/Users/hanbit-o/code/Visualization_wall_avoidance_results/Data/Result/ShaftInsertion/MHGP-BDI/0")
    # mhgp_bdi.demo_results(ID="2/20210908demo/20210908_181345demo/_random")
    # MB_mhgp_bdi = MHGP_BDI(
    #     dir_path="/Users/hanbit-o/code/Visualization_wall_avoidance_results/Data/Result/ShaftInsertion/20211223_BDI_shaft_insertion/Matsubara/Matsubara_MHGP-BDI/0")
    # MB_mhgp_bdi.test_results(ID="2/20211224_115249")

    BM_mhgp_bc = MHGP_BC(
        dir_path="/Users/hanbit-o/code/Visualization_wall_avoidance_results/Data/Result/ShaftInsertion/20211223_BDI_shaft_insertion/Brendan/Brendan_MGP-BC/0")
    BM_mhgp_bc.demo_results_v2(ID="0")
    BM_mhgp_bc.test_results(ID="0/20211220_145656")
