from datetime import datetime, timezone, timedelta
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import os
from os.path import isfile, join
import re
from utils import now_stamp

'''
goal:
    data -> figs
    figs -> video
'''


class VideoMaker():
    def __init__(self, video_name='video_name'):
        yd_string, time_string = now_stamp()
        dir_ID = video_name + '/' + yd_string
        self.figure_dir = 'Figures/' + dir_ID + '/' + time_string + '/'
        self.video_dir = 'Videos/' + dir_ID + '/'
        dirs = {'fig': self.figure_dir, 'video': self.video_dir}
        for key in dirs:
            if not os.path.exists(dirs[key]):
                os.makedirs(dirs[key])

        self.file_name = time_string

    # snapshots from learning process -------

    def get_plot(self, step, dataset, color='navy', alpha=1.0, linewidth=5):
        plt.style.use("ggplot")
        xx = torch.linspace(1, step + 1, steps=step)
        yy = dataset[:step]
        plt.plot(xx, yy, color=color, linewidth=linewidth, alpha=alpha)

    def get_figure(self, step):
        plt.xlim(0, 220)
        plt.ylim(0, 1.0)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        file_name = self.figure_dir + 'frame' + str(step)
        plt.savefig(file_name)
        plt.clf()
        # plt.show()

    def make_figs(self, main_dataset, sub_dataset=None, main_color='tomato', sub_alpha=1.0, linewidth=5, get_fig=None):
        if get_fig is None:
            os.error()
        if sub_dataset is not None:
            sub_plot = True
            sub_max_step = sub_dataset.shape[0]
        else:
            sub_plot = False

        max_step = main_dataset.shape[0]

        for i in range(max_step):
            if sub_plot:
                self.get_plot(sub_max_step, sub_dataset,
                              color='navy', alpha=sub_alpha, linewidth=linewidth)
            self.get_plot(i, main_dataset, color=main_color,
                          linewidth=linewidth)
            get_fig(i)
            done = int(i / (max_step - 1) * 10)
            bar = u"\u2588" * done + ' ' * (10 - done)
            per = (i + 1) * 100 / max_step
            print('\r[{}] {} %'.format(bar, per), end='')

    # figures to video ----------------------
    def tryint(self, s):
        try:
            return int(s)
        except:
            return s

    def alphanum_key(self, s):
        """ Turn a string into a list of string and number chunks.
            "z23a" -> ["z", 23, "a"]
        """
        return [self.tryint(c) for c in re.split('([0-9]+)', s)]

    def figure2video(self, fps=10):
        pathIn = self.figure_dir
        pathOut = self.video_dir + self.file_name + '.mp4'

        frame_array = []
        # files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f)) and ]
        files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))
                 and not f.startswith('.')]
        files.sort(key=self.alphanum_key)  # sorting the file names properly

        for i in range(len(files)):
            filename = pathIn + files[i]
            # reading each files
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width, height)

            # inserting the frames into an image array
            frame_array.append(img)
        # fourcc = cv2.VideoWriter_fourcc(*"MP4V")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(pathOut, fourcc, fps, size)
        # writing to a image array
        for i in range(len(frame_array)):
            out.write(frame_array[i])
        out.release()
        print('\nCompleted to make VIDEO: ' + self.file_name + '.mp4')


if __name__ == '__main__':
    # Import data set
    MGP = torch.load('Data/Disturbances/144719.pickle')
    MHGP = torch.load('Data/Disturbances/141223.pickle')

    videoMaker = VideoMaker(video_name='test')
    get_fig = videoMaker.get_figure
    # videoMaker.make_figs(main_dataset=MHGP, sub_dataset=MGP)
    videoMaker.make_figs(main_dataset=MHGP, get_fig=get_fig)
