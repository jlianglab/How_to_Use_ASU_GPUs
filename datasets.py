# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
import random
import copy
import csv



def build_dataset(is_train, args,mode="train",data_file=None):
    transform = build_transform(is_train)

    if args.data_set =="ChestXray":
        dataset =ChestX_ray14(args.data_path,data_file,transform, anno_percent=args.anno_percent)
        nb_classes =14
    return dataset, nb_classes


def build_transform(mode="train"):

    transformList = []
    normalize = transforms.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252])
    transCrop = 224
    transResize = 256
    if mode == "train":
        transformList.append(transforms.RandomResizedCrop(transCrop))
        transformList.append(transforms.RandomHorizontalFlip())
        transformList.append(transforms.RandomRotation(7))
        transformList.append(transforms.ToTensor())
        if normalize is not None:
            transformList.append(normalize)
    elif mode == "validation":
        transformList.append(transforms.Resize(transResize))
        transformList.append(transforms.CenterCrop(transCrop))
        transformList.append(transforms.ToTensor())
        if normalize is not None:
            transformList.append(normalize)
    elif mode == "test":
        transformList.append(transforms.Resize(transResize))
        transformList.append(transforms.TenCrop(transCrop))
        transformList.append(
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        if normalize is not None:
            transformList.append(
                transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
    transformSequence = transforms.Compose(transformList)

    return transformSequence




#==============================================NIH======================================================================
class ChestX_ray14(Dataset):

  def __init__(self, pathImageDirectory, pathDatasetFile, augment, num_class=14, anno_percent=100):

    self.img_list = []
    self.img_label = []
    self.augment = augment

    with open(pathDatasetFile, "r") as fileDescriptor:
      line = True

      while line:
        line = fileDescriptor.readline()

        if line:
          lineItems = line.split()

          imagePath = os.path.join(pathImageDirectory, lineItems[0])
          imageLabel = lineItems[1:num_class + 1]
          imageLabel = [int(i) for i in imageLabel]

          self.img_list.append(imagePath)
          self.img_label.append(imageLabel)

    indexes = np.arange(len(self.img_list))
    if anno_percent < 100:
      random.Random(99).shuffle(indexes)
      num_data = int(indexes.shape[0] * anno_percent / 100.0)
      indexes = indexes[:num_data]

      _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
      self.img_list = []
      self.img_label = []

      for i in indexes:
        self.img_list.append(_img_list[i])
        self.img_label.append(_img_label[i])

  def __getitem__(self, index):

    imagePath = self.img_list[index]

    imageData = Image.open(imagePath).convert('RGB')
    imageLabel = torch.FloatTensor(self.img_label[index])

    if self.augment != None: imageData = self.augment(imageData)

    return imageData, imageLabel

  def __len__(self):

    return len(self.img_list)
