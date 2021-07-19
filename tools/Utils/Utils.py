import os


def count_dirs_files(path):
    files = folders = 0

    for _, dirnames, filenames in os.walk(path):
        # ^ this idiom means "we won't be using this value"
        files += len(filenames)
        folders += len(dirnames)

    print("{:,} files, {:,} folders".format(files, folders))
    return files, folders


def count_dirs(path):
    folders = len(os.listdir(path))
    print("{:,} folders".format(folders))
    return folders


if __name__ == "__main__":
    # count_dirs_files("Data/20210609/45/MGP-HBDI")
    count_dirs("Data/Results/UGP-BC/")
