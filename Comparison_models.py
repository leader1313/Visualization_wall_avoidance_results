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
        self.success_rate['std'] = torch.stack(
            [results[i * 2:i * 2 + 2].mean() for i in range(int(results.shape[0] / 2))]).std()
        print(dataset.keys())

    def demo_results(self, ID):
        dataset = torch.load(self.dir_path + '/' + ID +
                             '/test_demo_results.pickle')
        self.success_rate = dataset['success_rate']
        results = torch.stack(dataset['test_results']).float() * 100
        self.success_rate['std'] = torch.stack(
            [results[i * 3:i * 3 + 3].mean() for i in range(int(results.shape[0] / 3))]).std()
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
