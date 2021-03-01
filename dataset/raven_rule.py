# -*- coding: utf-8 -*-


import glob
import os

import cv2
import numpy as np
import torch
import torch.utils.data.dataset
from PIL import Image


def imresize(img_src, size):
    return np.array(Image.fromarray(img_src).resize(size=size))


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, dataset_type, img_size, config, test=False, shuffle=False):
        self.dataset_path = dataset_path
        self.file_names = [f for f in glob.glob(os.path.join(self.dataset_path, config, "*.npz")) \
                           if dataset_type in f and "rule" not in f and "attr" not in f]
        self.img_size = img_size
        self.shuffle = shuffle
        self.config = config
        self.second_component = None
        self.test = test

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        data_path = self.file_names[idx]
        data = np.load(data_path)
        image = data["image"]
        target = data["target"]

        if self.shuffle:
            context = image[:8, :, :]
            choices = image[8:, :, :]
            indices = range(8)
            np.random.shuffle(indices)
            target = indices.index(target)
            choices = choices[indices, :, :]
            image = np.concatenate((context, choices))

        image = getattr(self, self.config)(image)

        image = torch.tensor(image, dtype=torch.float)
        target = torch.tensor(target, dtype=torch.long)

        if self.test:
            rule_ret = 0
        else:
            rule_gt = np.load(data_path.replace(".npz", "_rule.npz"))
            pos_num_rule = torch.tensor(rule_gt["pos_num_rule"], dtype=torch.long)
            type_rule = torch.tensor(rule_gt["type_rule"], dtype=torch.long)
            size_rule = torch.tensor(rule_gt["size_rule"], dtype=torch.long)
            color_rule = torch.tensor(rule_gt["color_rule"], dtype=torch.long)
            rule_ret = (pos_num_rule, type_rule, size_rule, color_rule)
        return image, target, rule_ret

    def center_single(self, image):
        # image: 16, 160, 160
        resize_image = []
        for idx in range(16):
            resize_image.append(imresize(image[idx, :, :], (self.img_size, self.img_size)))
        image = np.stack(resize_image)[:, np.newaxis, :, :]
        return image

    def distribute_four(self, image):
        # image: 16, 160, 160
        part_1 = image[:, :80, :80]
        part_2 = image[:, :80, 80:160]
        part_3 = image[:, 80:160, :80]
        part_4 = image[:, 80:160, 80:160]
        resize_part_1 = []
        resize_part_2 = []
        resize_part_3 = []
        resize_part_4 = []
        for idx in range(16):
            resize_part_1.append(imresize(part_1[idx, :, :], (self.img_size, self.img_size)))
            resize_part_2.append(imresize(part_2[idx, :, :], (self.img_size, self.img_size)))
            resize_part_3.append(imresize(part_3[idx, :, :], (self.img_size, self.img_size)))
            resize_part_4.append(imresize(part_4[idx, :, :], (self.img_size, self.img_size)))
        resize_part_1 = np.stack(resize_part_1)
        resize_part_2 = np.stack(resize_part_2)
        resize_part_3 = np.stack(resize_part_3)
        resize_part_4 = np.stack(resize_part_4)
        image = np.stack([resize_part_1, resize_part_2, resize_part_3, resize_part_4], axis=1)
        return image

    def distribute_nine(self, image):
        # image: 16, 160, 160
        part_1 = image[:, :53, :53]
        part_2 = image[:, :53, 53:106]
        part_3 = image[:, :53, 106:160]
        part_4 = image[:, 53:106, :53]
        part_5 = image[:, 53:106, 53:106]
        part_6 = image[:, 53:106, 106:160]
        part_7 = image[:, 106:160, :53]
        part_8 = image[:, 106:160, 53:106]
        part_9 = image[:, 106:160, 106:160]
        resize_part_1 = []
        resize_part_2 = []
        resize_part_3 = []
        resize_part_4 = []
        resize_part_5 = []
        resize_part_6 = []
        resize_part_7 = []
        resize_part_8 = []
        resize_part_9 = []
        for idx in range(16):
            resize_part_1.append(imresize(part_1[idx, :, :], (self.img_size, self.img_size)))
            resize_part_2.append(imresize(part_2[idx, :, :], (self.img_size, self.img_size)))
            resize_part_3.append(imresize(part_3[idx, :, :], (self.img_size, self.img_size)))
            resize_part_4.append(imresize(part_4[idx, :, :], (self.img_size, self.img_size)))
            resize_part_5.append(imresize(part_5[idx, :, :], (self.img_size, self.img_size)))
            resize_part_6.append(imresize(part_6[idx, :, :], (self.img_size, self.img_size)))
            resize_part_7.append(imresize(part_7[idx, :, :], (self.img_size, self.img_size)))
            resize_part_8.append(imresize(part_8[idx, :, :], (self.img_size, self.img_size)))
            resize_part_9.append(imresize(part_9[idx, :, :], (self.img_size, self.img_size)))
        resize_part_1 = np.stack(resize_part_1)
        resize_part_2 = np.stack(resize_part_2)
        resize_part_3 = np.stack(resize_part_3)
        resize_part_4 = np.stack(resize_part_4)
        resize_part_5 = np.stack(resize_part_5)
        resize_part_6 = np.stack(resize_part_6)
        resize_part_7 = np.stack(resize_part_7)
        resize_part_8 = np.stack(resize_part_8)
        resize_part_9 = np.stack(resize_part_9)
        image = np.stack([resize_part_1, resize_part_2, resize_part_3, resize_part_4, resize_part_5, resize_part_6, resize_part_7, resize_part_8, resize_part_9], axis=1)
        return image

    def in_center_single_out_center_single(self, image):
        # image: 16, 160, 160
        part_1 = image[:, 53:106, 53:106]
        part_2 = np.copy(image[:, :, :])
        part_2[:, 53:106, 53:106] = 255
        resize_part_1 = []
        resize_part_2 = []
        for idx in range(16):
            resize_part_1.append(imresize(part_1[idx, :, :], (self.img_size, self.img_size)))
            resize_part_2.append(imresize(part_2[idx, :, :], (self.img_size, self.img_size)))
        resize_part_1 = np.stack(resize_part_1)
        resize_part_2 = np.stack(resize_part_2)
        image = np.stack([resize_part_1, resize_part_2], axis=1)
        self.second_component = 1
        return image

    def in_distribute_four_out_center_single(self, image):
        # image: 16, 160, 160
        part_1 = image[:, 55:79, 55:79]
        part_2 = image[:, 55:79, 81:105]
        part_3 = image[:, 81:105, 55:79]
        part_4 = image[:, 81:105, 81:105]
        part_5 = np.copy(image[:, :, :])
        part_5[:, 55:79, 55:79] = 255
        part_5[:, 55:79, 81:105] = 255
        part_5[:, 81:105, 55:79] = 255
        part_5[:, 81:105, 81:105] = 255
        resize_part_1 = []
        resize_part_2 = []
        resize_part_3 = []
        resize_part_4 = []
        resize_part_5 = []
        kernel = np.ones((3, 3), np.uint8)
        for idx in range(16):
            resize_part_1.append(cv2.dilate(imresize(part_1[idx, :, :], (self.img_size, self.img_size)), kernel))
            resize_part_2.append(cv2.dilate(imresize(part_2[idx, :, :], (self.img_size, self.img_size)), kernel))
            resize_part_3.append(cv2.dilate(imresize(part_3[idx, :, :], (self.img_size, self.img_size)), kernel))
            resize_part_4.append(cv2.dilate(imresize(part_4[idx, :, :], (self.img_size, self.img_size)), kernel))
            resize_part_5.append(imresize(part_5[idx, :, :], (self.img_size, self.img_size)))
        resize_part_1 = np.stack(resize_part_1)
        resize_part_2 = np.stack(resize_part_2)
        resize_part_3 = np.stack(resize_part_3)
        resize_part_4 = np.stack(resize_part_4)
        resize_part_5 = np.stack(resize_part_5)
        image = np.stack([resize_part_1, resize_part_2, resize_part_3, resize_part_4, resize_part_5], axis=1)
        self.second_component = 4
        return image

    def left_center_single_right_center_single(self, image):
        # image: 16, 160, 160
        part_1 = image[:, 40:120, :80]
        part_2 = image[:, 40:120, 81:160]
        resize_part_1 = []
        resize_part_2 = []
        for idx in range(16):
            resize_part_1.append(imresize(part_1[idx, :, :], (self.img_size, self.img_size)))
            resize_part_2.append(imresize(part_2[idx, :, :], (self.img_size, self.img_size)))
        resize_part_1 = np.stack(resize_part_1)
        resize_part_2 = np.stack(resize_part_2)
        image = np.stack([resize_part_1, resize_part_2], axis=1)
        self.second_component = 1
        return image

    def up_center_single_down_center_single(self, image):
        # image: 16, 160, 160
        part_1 = image[:, :80, 40:120]
        part_2 = image[:, 81:160, 40:120]
        resize_part_1 = []
        resize_part_2 = []
        for idx in range(16):
            resize_part_1.append(imresize(part_1[idx, :, :], (self.img_size, self.img_size)))
            resize_part_2.append(imresize(part_2[idx, :, :], (self.img_size, self.img_size)))
        resize_part_1 = np.stack(resize_part_1)
        resize_part_2 = np.stack(resize_part_2)
        image = np.stack([resize_part_1, resize_part_2], axis=1)
        self.second_component = 1
        return image


class DatasetAttr(Dataset):
    # Shape:
    # num_of_obj: {
    #     "center_single", 1
    #     "distribute_four", 4,
    #     "distribute_nine", 9,
    #     "left_center_single_right_center_single", 2,
    #     "up_center_single_down_center_single", 2,
    #     "in_center_single_out_center_single", 2,
    #     "in_distribute_four_out_center_single", 5,
    # }
    # exist: (B, num_of_obj) 0-1
    # type: (B, num_of_obj) 0-4 (if exist=0, type=4)
    # size: (B, num_of_obj) 0-5 (if exist=0, size=5)
    # color: (B, num_of_obj) 0-9 (if exist=0, color=9)
    def __init__(self, dataset_path, dataset_type, img_size, config, shuffle=False):

        self.dataset_path = dataset_path
        self.file_names = [f for f in glob.glob(os.path.join(self.dataset_path, config, "*.npz")) \
                           if dataset_type in f and "rule" not in f and "attr" not in f]
        self.img_size = img_size
        self.shuffle = shuffle
        self.config = config
        self.second_component = None

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        data_path = self.file_names[idx]
        data = np.load(data_path)
        image = data["image"]
        target = data["target"]

        if self.shuffle:
            context = image[:8, :, :]
            choices = image[8:, :, :]
            indices = range(8)
            np.random.shuffle(indices)
            target = indices.index(target)
            choices = choices[indices, :, :]
            image = np.concatenate((context, choices))

        image = getattr(self, self.config)(image)

        image = torch.tensor(image, dtype=torch.float)
        attr_gt = np.load(data_path.replace(".npz", "_attr.npz"))
        exist_gt = torch.tensor(attr_gt["exist"], dtype=torch.long)
        type_gt = torch.tensor(attr_gt["type"], dtype=torch.long)
        size_gt = torch.tensor(attr_gt["size"], dtype=torch.long)
        color_gt = torch.tensor(attr_gt["color"], dtype=torch.long)

        return image, target, exist_gt, type_gt, size_gt, color_gt


class DatasetRule(Dataset):
    # Shape:
    # pos_num_rule: (B) 0-12 (2x2) 0-16 (3x3) 0 (others)
    # type_rule: (B) 0-6
    # size_rule: (B) 0-6
    # color_rule: (B) 0-8
    def __init__(self, dataset_path, dataset_type, img_size, config, shuffle=False):
        if "left" in config or "up" in config or "out" in config:
            raise NotImplementedError("Only support one comp config")
        self.dataset_path = dataset_path
        self.file_names = [f for f in glob.glob(os.path.join(self.dataset_path, config, "*.npz")) \
                           if dataset_type in f and "rule" not in f and "attr" not in f]
        self.img_size = img_size
        self.shuffle = shuffle
        self.config = config
        self.second_component = None

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        data_path = self.file_names[idx]
        data = np.load(data_path)
        image = data["image"]
        target = data["target"]

        if self.shuffle:
            context = image[:8, :, :]
            choices = image[8:, :, :]
            indices = range(8)
            np.random.shuffle(indices)
            target = indices.index(target)
            choices = choices[indices, :, :]
            image = np.concatenate((context, choices))

        image = getattr(self, self.config)(image)

        image = torch.tensor(image, dtype=torch.float)
        rule_gt = np.load(data_path.replace(".npz", "_rule_comp0.npz"))
        pos_num_rule = torch.tensor(rule_gt["pos_num_rule"], dtype=torch.long)
        type_rule = torch.tensor(rule_gt["type_rule"], dtype=torch.long)
        size_rule = torch.tensor(rule_gt["size_rule"], dtype=torch.long)
        color_rule = torch.tensor(rule_gt["color_rule"], dtype=torch.long)

        return image, target, pos_num_rule, type_rule, size_rule, color_rule


class DatasetGT(DatasetzR):
    def __init__(self, dataset_path, dataset_type, img_size, config, shuffle=False):
        self.dataset_path = dataset_path
        self.file_names = [f for f in glob.glob(os.path.join(self.dataset_path, config, "*.npz")) \
                           if dataset_type in f and "rule" not in f and "attr" not in f]
        self.img_size = img_size
        self.shuffle = shuffle
        self.config = config
        self.second_component = None

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        data_path = self.file_names[idx]
        data = np.load(data_path)
        image = data["image"]
        target = data["target"]

        if self.shuffle:
            context = image[:8, :, :]
            choices = image[8:, :, :]
            indices = range(8)
            np.random.shuffle(indices)
            target = indices.index(target)
            choices = choices[indices, :, :]
            image = np.concatenate((context, choices))


        attr_gt = np.load(data_path.replace(".npz", "_attr.npz"))
        exist_gt = torch.tensor(attr_gt["exist"], dtype=torch.long)
        type_gt = torch.tensor(attr_gt["type"], dtype=torch.long)
        size_gt = torch.tensor(attr_gt["size"], dtype=torch.long)
        color_gt = torch.tensor(attr_gt["color"], dtype=torch.long)
        target = torch.tensor(target, dtype=torch.long)

        return exist_gt, type_gt, size_gt, color_gt, target
