import torch


class ComparisonModels():
    def __init__(self, dir_path, name, color):
        self.dir_path = dir_path
        self.color = color
        self.name = name

    def test_results(self, ID):
        dataset = torch.load(
            self.dir_path + '/' + ID + '/_random/test_results.pickle')
        self.success_rate = dataset['success_rate']
        results = torch.stack(
            dataset['test_results']).float() * 100
        interbal = 5
        self.success_rate['std'] = torch.stack(
            [results[i * interbal:i * interbal + interbal].mean() for i in range(int(results.shape[0] / interbal))]).std()
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
        name = "MHGP-BC"
        color = "navy"
        super().__init__(dir_path, name, color)


class UHGP_BC(ComparisonModels):
    def __init__(self, dir_path):
        name = "UHGP-BDI"
        color = "grey"
        super().__init__(dir_path, name, color)


if __name__ == "__main__":
    mhgp_bdi = MHGP_BDI(
        dir_path="/Users/hanbit-o/code/Visualization_wall_avoidance_results/Data/Result/ShaftInsertion/MHGP-BDI/0")
    mhgp_bdi.demo_results(ID="2/20210908demo/20210908_181345demo/_random")
