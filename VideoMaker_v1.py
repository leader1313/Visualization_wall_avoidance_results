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
    def __init__(self, dataset=None, video_name='video_name'):
        yd_string, time_string = now_stamp()
        dir_ID = video_name + '/' + yd_string
        self.figure_dir = 'Figures/' + dir_ID + '/' + time_string + '/'
        self.video_dir = 'Videos/' + dir_ID + '/'
        dirs = {'fig': self.figure_dir, 'video': self.video_dir}
        for key in dirs:
            if not os.path.exists(dirs[key]):
                os.makedirs(dirs[key])

        self.file_name = time_string
        self.dataset = dataset

        self.max_steps = self.dataset.shape[0]

    # snapshots from learning process -------

    def get_hetero_disturbance_figure(self, step, dir=None):
        plt.style.use("ggplot")

        # state-independent disturbance
        fail_max_step = 110
        xx = torch.linspace(1, fail_max_step, steps=fail_max_step)
        yy = torch.ones(fail_max_step) * 0.41
        baseline = plt.plot(xx, yy, color='navy', linewidth=5)

        # state-dependent disturbance
        injected_disturbances = self.dataset

        X = torch.linspace(1, step + 1, steps=step + 1)
        Y = injected_disturbances[:step + 1]

        # plt.xlim(0, self.dataset.shape[0])
        plt.xlim(0, 220)
        plt.ylim(0, 1.0)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        color = "tomato"
        line = plt.plot(X, Y, color=color, linewidth=5)

        file_name = dir + 'frame' + str(step)
        plt.savefig(file_name)
        plt.clf()

    def get_homo_disturbance_figure(self, step, dir=None):
        plt.style.use("ggplot")

        # state-independent disturbance

        injected_disturbances = self.dataset

        X = torch.linspace(1, step + 1, steps=step + 1)
        Y = injected_disturbances[:step + 1]

        # plt.xlim(0, self.dataset.shape[0])
        plt.xlim(0, 220)
        plt.ylim(0, 1.0)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        color = "navy"
        line = plt.plot(X, Y, color=color, linewidth=5)

        file_name = dir + 'frame' + str(step)
        plt.savefig(file_name)
        plt.clf()

    def make_figs(self, get_figs=None):
        if get_figs is None:
            os.error
        max_steps = self.max_steps
        for i in range(max_steps):
            get_figs(i, dir=self.figure_dir)
            done = int(i / (max_steps - 1) * 10)
            bar = u"\u2588" * done + ' ' * (10 - done)
            per = (i + 1) * 100 / max_steps
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
    # MGP-BDI
    dataset = torch.load('Data/Disturbances/144719.pickle')

    # Video making
    videoMaker = VideoMaker(dataset=dataset, video_name='test')
    videoMaker.make_figs(get_figs=videoMaker.get_homo_disturbance_figure)
    videoMaker.figure2video()

    # MHGP
    dataset = torch.load('Data/Disturbances/141223.pickle')

    # Video making
    videoMaker = VideoMaker(dataset=dataset, video_name='test')
    videoMaker.make_figs(get_figs=videoMaker.get_hetero_disturbance_figure)
    videoMaker.figure2video()
