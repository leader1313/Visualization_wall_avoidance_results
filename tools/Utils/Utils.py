import os
import torch


def count_dirs_files(path):
    files = folders = 0

    for _, dirnames, filenames in os.walk(path):
        # ^ this idiom means "we won't be using this value"
        files += len(filenames)
        folders += len(dirnames)

    print("{:,} files, {:,} folders".format(files, folders))
    return files, folders


def point_approximation(dataset):
    N = dataset.shape[0]
    interbal = torch.tensor(N).sqrt().int()
    means = torch.stack(
        [dataset[i: i + interbal].mean() for i in range(int(N / interbal))]
    )
    mean = dataset.mean()
    std = means.std()
    return mean, std


def count_dirs(path):
    folders = len(os.listdir(path))
    print("{:,} folders".format(folders))
    return folders


if __name__ == "__main__":
    # count_dirs_files("Data/20210609/45/MGP-HBDI")
    count_dirs("Data/Results/UGP-BC/")
