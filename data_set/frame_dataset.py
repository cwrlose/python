import torch.utils.data as data
import os
import os.path
import numpy as np
from numpy.random import randint
import pdb
import io
import time
import pandas as pd
import torchvision
import random
from PIL import Image, ImageOps
import cv2
import numbers
import math
import torch
from RandAugment import RandAugment


"""
self, list_file, labels_file,image_tmpl='img_%05d.jpg,index_bias=1 :根据文件名字来
num_segments=1, new_length=1,:将视频帧数分段 每段多少帧数
transform=None,random_shift=True, test_mode=False,
返回一个([video_frame],labels)
"""

"""
val_data = Action_DATASETS(config.data.val_list, config.data.label_list, num_segments=config.data.num_segments,
                        image_tmpl=config.data.image_tmpl,
                        transform=transform_val, random_shift=config.random_shift)
"""
"""
该类用于对一组图像(img_group)中的每一张图像应用相同的变换操作。在初始化时,接收一个变换函数 
transform 并将其存储在 self.worker 中。当调用该类的实例时，
会遍历图像组中的每一张图像，并对其应用 self.worker 所代表的变换，最后返回变换后的图像组
"""
class GroupTransform(object):
    def __init__(self, transform):
        self.worker = transform

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]#推导式
"""
归一化
"""
    
class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255) if self.div else img.float()

"""
该类用于将一组图像堆叠在一起
"""

class Stack(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == 'L':
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                print(len(img_group))
                return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                print(len(img_group))
                rst = np.concatenate(img_group, axis=2)
                return rst

"""
类用于封装视频记录的信息，包括视频的路径、帧数和标签。在初始化时，
接收一个包含这些信息的列表 row,并通过属性方法 path、num_frames 和 label 来访问这些信息
""" 
class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])

""""
__getitem__方法:当通过索引访问DATASETS实例时,__getitem__方法会被调用。
它首先根据random_shift的值来确定获取视频帧索引的方式,调用_sample_indices方法或_get_val_indices方法得到segment_indices。然后调用get方法,
传入record和segment_indices。
get方法:该方法遍历segment_indices中的每个索引,通过_load_image方法加载对应索引的图像帧,将这些图像帧存储在images列表中。最后,通过self.transform(images)对图像列表进行处理，返回处理后的图像数据process_data以及record.label，即对应的标签。因此，DATASETS返回的是经过一系列处理后的多帧图像数据和与之对应的一个标签，
不是单纯的一帧图像和一个标签。
""""



class DATASETS(data.Dataset):
    def __init__(self, list_file, labels_file,
                 num_segments=1, new_length=1,
                 image_tmpl='img_%05d.jpg', transform=None,
                 random_shift=True, test_mode=False, index_bias=1):

        self.list_file = list_file
        self.num_segments = num_segments
        self.seg_length = new_length
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.loop=False
        self.index_bias = index_bias
        self.labels_file = labels_file

        if self.index_bias is None:
            if self.image_tmpl == "frame{:d}.jpg":
                self.index_bias = 0
            else:
                self.index_bias = 1
        self._parse_list()
        self.initialized = False

    def _load_image(self, directory, idx):

        return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]
    @property
    def total_length(self):
        return self.num_segments * self.seg_length
    
    @property
    def classes(self):
        classes_all = pd.read_csv(self.labels_file)
        return classes_all.values.tolist()
    
    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]

    def _sample_indices(self, record):
        if record.num_frames <= self.total_length:
            if self.loop:
                return np.mod(np.arange(
                    self.total_length) + randint(record.num_frames // 2),
                    record.num_frames) + self.index_bias
            offsets = np.concatenate((
                np.arange(record.num_frames),
                randint(record.num_frames,
                        size=self.total_length - record.num_frames)))
            return np.sort(offsets) + self.index_bias
        offsets = list()
        ticks = [i * record.num_frames // self.num_segments
                 for i in range(self.num_segments + 1)]

        for i in range(self.num_segments):
            tick_len = ticks[i + 1] - ticks[i]
            tick = ticks[i]
            if tick_len >= self.seg_length:
                tick += randint(tick_len - self.seg_length + 1)
            offsets.extend([j for j in range(tick, tick + self.seg_length)])
        return np.array(offsets) + self.index_bias

    def _get_val_indices(self, record):
        if self.num_segments == 1:
            return np.array([record.num_frames //2], dtype=np.int) + self.index_bias
        
        if record.num_frames <= self.total_length:
            if self.loop:
                return np.mod(np.arange(self.total_length), record.num_frames) + self.index_bias
            return np.array([i * record.num_frames // self.total_length
                             for i in range(self.total_length)], dtype=np.int) + self.index_bias
        offset = (record.num_frames / self.num_segments - self.seg_length) / 2.0
        return np.array([i * record.num_frames / self.num_segments + offset + j
                         for i in range(self.num_segments)
                         for j in range(self.seg_length)], dtype=np.int) + self.index_bias

    def __getitem__(self, index):
        record = self.video_list[index]
        segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        return self.get(record, segment_indices)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


    def get(self, record, indices):
        images = list()
        for i, seg_ind in enumerate(indices):
            p = int(seg_ind)
            try:
                seg_imgs = self._load_image(record.path, p)
            except OSError:
                print('ERROR: Could not read image "{}"'.format(record.path))
                print('invalid indices: {}'.format(indices))
                raise
            images.extend(seg_imgs)
        process_data = self.transform(images)
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)

