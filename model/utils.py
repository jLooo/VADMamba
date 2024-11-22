import numpy as np
from collections import OrderedDict
import os
import glob
import cv2
import torch.utils.data as data
import re

rng = np.random.RandomState(2020)


def np_load_frame(filename, resize_height, resize_width):
    image_decoded = cv2.imread(filename)
    image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
    image_resized = image_resized.astype(dtype=np.float32)
    image_resized = (image_resized / 127.5) - 1.0
    return image_resized


class DataLoader(data.Dataset):
    def __init__(self, video_folder, transform, resize_height, resize_width, train=True, time_step=4, num_pred=1):
        self.dir = video_folder
        self.transform = transform
        self.videos = OrderedDict()
        self._resize_height = resize_height
        self._resize_width = resize_width
        self._time_step = time_step
        self._num_pred = num_pred
        self.train = train
        self.setup()
        self.samples = self.get_all_samples()

    def setup(self):
        videos = glob.glob(os.path.join(self.dir, '*'))
        for idx, video in enumerate(sorted(videos)):
            video_name = video.split('/')[-1]
            # dataset_name = video.split('/')[6]  # # debug
            format = '*.jpg'  # '*.tif' if dataset_name == 'ped1' else '*.jpg'
            self.videos[video_name] = {}
            self.videos[video_name]['path'] = video
            self.videos[video_name]['frame'] = glob.glob(os.path.join(video, format))
            self.videos[video_name]['frame'].sort()
            self.videos[video_name]['idx'] = idx
            self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])

    def get_all_samples(self):
        frames = []
        videos = glob.glob(os.path.join(self.dir, '*'))

        for video in sorted(videos):
            video_name = video.split('/')[-1]
            for i in range(len(self.videos[video_name]['frame']) - self._time_step):
                frames.append(self.videos[video_name]['frame'][i])

        return frames

    def __getitem__(self, index):
        video_name = self.samples[index].split('/')[-2]
        frame_name = int(self.samples[index].split('.')[-2].split('/')[-1])
        batch_forward = []

        for i in range(self._time_step + self._num_pred):
            image_f = np_load_frame(self.videos[video_name]['frame'][frame_name + i], self._resize_height,
                                    self._resize_width)

            if self.transform is not None:
                batch_forward.append(self.transform(image_f))

        return np.concatenate(batch_forward, axis=0)

    def __len__(self):
        return len(self.samples)
