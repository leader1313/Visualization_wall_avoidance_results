from datetime import datetime, timezone, timedelta
from natsort import natsorted
import pickle
import torch
import os


class Repository:
    """Repository tree
    Data-
        ├── History
        ├── Figures
            ├── performance
        ├── today (20210512)
            ├── 0 (# of experiment)
                ├── title (BC/ BDI)
                    ├── 0 (# of Trial)
                        ├── 0 (# of iter)
                            ├── learner.pickle
                            └── results.pickle
    """

    def __init__(self, dirmake=True):
        # datetime object containing current date and time
        now = datetime.now(timezone(timedelta(hours=9)))
        self.d_string = now.strftime("%Y%m%d")
        self.dir_path = "Data/" + self.d_string
        if dirmake:
            # results ---------------------------
            if not os.path.exists(self.dir_path + "/0/"):
                self.dir_path = self.dir_path + "/0/"
            else:
                dir_list = natsorted(os.listdir(self.dir_path))
                dir_list = [int(i) for i in dir_list]
                self.dir_path = self.dir_path + \
                    "/" + str(max(dir_list) + 1) + "/"
            os.makedirs(self.dir_path)
            # histories -------------------------
            self.dir_history = "Data/History/"
            if not os.path.exists(self.dir_history):
                os.makedirs(self.dir_history)
            pass
            # figures -------------------------
            self.dir_figures = "Data/Figures/"
            if not os.path.exists(self.dir_figures):
                os.makedirs(self.dir_figures)
            pass

    def dt_string(self):
        now = datetime.now(timezone(timedelta(hours=9)))
        dt_string = now.strftime("%Y%m%d_%H%M%S")
        return dt_string

    def save_data(self, data, dir_path=None, file_name=None):
        # history save -----------------
        # torch.save(
        #     data, self.dir_history + self.dt_string() + ".pickle", pickle_module=pickle
        # )

        # optional save ----------------
        # 'title/iteration/learner'
        try:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            path = dir_path + file_name + ".pickle"
            torch.save(data, path, pickle_module=pickle)
            print(path)
        except:
            pass

    def load_data(self, dir_path):
        data = torch.load(dir_path)
        return data


if __name__ == "__main__":
    dir_path = "dd"
    data = "dd"
    repo = Repository()
    repo.save_data(data=data)
