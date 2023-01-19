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


class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


def build_dataset(is_train, args,mode="train",data_file=None):
    transform = build_transform(is_train)

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set =="ChestXray":
        # data =ChestX_ray14(args.data_path,transform,is_train,args.anno_percent,mode=mode,split=args.split)
        dataset =ChestX_ray14(args.data_path,data_file,transform, anno_percent=args.anno_percent)
        nb_classes =14
    elif args.data_set =="ChexPert":
        dataset =CheXpert(args.data_path,data_file,transform,anno_percent=args.anno_percent)
        nb_classes =14
    elif args.data_set =="VinDrCXRGlobalOne" or args.data_set =="VinDrCXRGlobalVote":
        dataset =VinDrCXR(args.data_path,data_file,transform,anno_percent=args.anno_percent)
        nb_classes =6
    elif args.data_set == "VinDrCXRLocalOne" or args.data_set == "VinDrCXRLocalVote":
        dataset = VinDrCXR(args.data_path, data_file, transform, anno_percent=args.anno_percent)
        nb_classes = 22
    elif args.data_set =="RSNAPneumonia":
        dataset =RSNAPneumonia(args.data_path,data_file,transform,annotation_percent=args.anno_percent)
        nb_classes =3
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
    elif mode == "test_no_ten":
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



def build_rotation_transform(mode="train"):

    transformList = []
    normalize = transforms.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252])
    transCrop = 448
    transResize = 576
    if mode == "train":
        transformList.append(transforms.Resize(transResize))
        transformList.append(transforms.CenterCrop(transCrop))
        transformList.append(transforms.ToTensor())

        if normalize is not None:
            transformList.append(normalize)
    elif mode == "validation":
        transformList.append(transforms.Resize(transResize))
        transformList.append(transforms.CenterCrop(transCrop))
        transformList.append(transforms.ToTensor())

        if normalize is not None:
            transformList.append(normalize)
    else:
        transformList.append(transforms.Resize(transResize))
        transformList.append(transforms.CenterCrop(transCrop))
        transformList.append(transforms.ToTensor())


        if normalize is not None:
            transformList.append(normalize)

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



class ChestX_ray14_rotation(Dataset):

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

    rotate_prob = random.random()

    if rotate_prob <0.33:
        rotate_label = torch.FloatTensor([0.0])
    elif rotate_prob>=0.33 and rotate_prob <=0.66:
        imageData  = imageData.rotate(90, Image.NEAREST, expand = 1)
        rotate_label = torch.FloatTensor([1.0])
    else:
        imageData = imageData.rotate(-90, Image.NEAREST, expand = 1)
        rotate_label = torch.FloatTensor([1.0])


    if self.augment != None: imageData = self.augment(imageData)

    return imageData, rotate_label

  def __len__(self):

    return len(self.img_list)



#==============================================CheXpert======================================================================


class CheXpert(Dataset):

  def __init__(self, pathImageDirectory, pathDatasetFile, augment, num_class=14,
               uncertain_label="LSR-Ones", unknown_label=0, anno_percent=100):

    self.img_list = []
    self.img_label = []
    self.augment = augment
    assert uncertain_label in ["Ones", "Zeros", "LSR-Ones", "LSR-Zeros"]
    self.uncertain_label = uncertain_label

    with open(pathDatasetFile, "r") as fileDescriptor:
      csvReader = csv.reader(fileDescriptor)
      next(csvReader, None)
      for line in csvReader:
        imagePath = os.path.join(pathImageDirectory, line[0])
        label = line[5:]
        for i in range(num_class):
          if label[i]:
            a = float(label[i])
            if a == 1:
              label[i] = 1
            elif a == 0:
              label[i] = 0
            elif a == -1: # uncertain label
              label[i] = -1
          else:
            label[i] = unknown_label # unknown label

        self.img_list.append(imagePath)
        imageLabel = [int(i) for i in label]
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

    label = []
    for l in self.img_label[index]:
      if l == -1:
        if self.uncertain_label == "Ones":
          label.append(1)
        elif self.uncertain_label == "Zeros":
          label.append(0)
        elif self.uncertain_label == "LSR-Ones":
          label.append(random.uniform(0.55, 0.85))
        elif self.uncertain_label == "LSR-Zeros":
          label.append(random.uniform(0, 0.3))
      else:
        label.append(l)
    imageLabel = torch.FloatTensor(label)

    if self.augment != None: imageData = self.augment(imageData)

    return imageData, imageLabel

  def __len__(self):

    return len(self.img_list)



#==============================================VinDrCXR======================================================================
class VinDrCXR(Dataset):
    def __init__(self, pathImageDirectory, pathDatasetFile, augment, anno_percent=100):
        self.img_list = []
        self.img_label = []
        self.augment = augment

        with open(pathDatasetFile, "r") as fr:
            line = fr.readline().strip()
            while line:
                lineItems = line.split()
                imagePath = os.path.join(pathImageDirectory, lineItems[0]+".jpeg")
                imageLabel = lineItems[1:]
                imageLabel = [int(i) for i in imageLabel]
                self.img_list.append(imagePath)
                self.img_label.append(imageLabel)
                line = fr.readline()

        if anno_percent < 100:
            indexes = np.arange(len(self.img_list))
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
        imageLabel = torch.FloatTensor(self.img_label[index])
        imageData = Image.open(imagePath).convert('RGB')
        if self.augment != None: imageData = self.augment(imageData)
        return imageData, imageLabel
    def __len__(self):

        return len(self.img_list)



class ShenzhenCXR(Dataset):

  def __init__(self, images_path, file_path, augment, num_class=1, annotation_percent=100):

    self.img_list = []
    self.img_label = []
    self.augment = augment

    with open(file_path, "r") as fileDescriptor:
      line = True

      while line:
        line = fileDescriptor.readline()
        if line:
          lineItems = line.split(',')

          imagePath = os.path.join(images_path, lineItems[0])
          imageLabel = lineItems[1:num_class + 1]
          imageLabel = [int(i) for i in imageLabel]

          self.img_list.append(imagePath)
          self.img_label.append(imageLabel)

    indexes = np.arange(len(self.img_list))
    if annotation_percent < 100:
      random.Random(99).shuffle(indexes)
      num_data = int(indexes.shape[0] * annotation_percent / 100.0)
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


#==============================================RSNAPneumonia======================================================================

class RSNAPneumonia(Dataset):

  def __init__(self, images_path, file_path, augment, annotation_percent=100):

    self.img_list = []
    self.img_label = []
    self.augment = augment

    with open(file_path, "r") as fileDescriptor:
      line = True

      while line:
        line = fileDescriptor.readline()
        if line:
          lineItems = line.strip().split(' ')
          imagePath = os.path.join(images_path, lineItems[0])


          self.img_list.append(imagePath)
          self.img_label.append(int(lineItems[-1]))

    indexes = np.arange(len(self.img_list))
    if annotation_percent < 100:
      random.Random(99).shuffle(indexes)
      num_data = int(indexes.shape[0] * annotation_percent / 100.0)
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
    imageLabel = self.img_label[index]
    if self.augment != None: imageData = self.augment(imageData)

    return imageData, imageLabel

  def __len__(self):

    return len(self.img_list)