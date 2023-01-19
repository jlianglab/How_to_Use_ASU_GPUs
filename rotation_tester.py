import torch.nn as nn
import torch
import argparse
import os
from config import Nih14 as config
from datasets import ChestX_ray14_rotation,build_rotation_transform
from networks import resnet50
import time
from utils import AverageMeter,save_model, computeAUROC
import numpy as np
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys
from torch.utils.data import Dataset
from os.path import isfile, join
from os import listdir
from PIL import Image
from torchvision import transforms

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

main_path = "/mnt/dfs/jpang12/datasets/PLCOI-880"
sub_dict = ["batch_1a","batch_1b","batch_1c","batch_1d","batch_1e","batch_1f",
            "batch_2a","batch_2b","batch_2c","batch_2d","batch_2e","batch_2f"]
weight = "/mnt/dfs/jpang12/downstream_2d/saved_models/nih14_resnet50_run3_angle_rotation/ckpt_epoch_50.pth"
batch_size = 256
class Rot_dataset(Dataset):

  def __init__(self):
    self.img_list = []
    main_path = "/mnt/dfs/jpang12/datasets/PLCOI-880"
    sub_dict = ["batch_1a", "batch_1b", "batch_1c", "batch_1d", "batch_1e", "batch_1f",
                "batch_2a", "batch_2b", "batch_2c", "batch_2d", "batch_2e", "batch_2f"]

    for dict in sub_dict:
        mypath = join(main_path, dict)
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        for file in onlyfiles:
            self.img_list.append(join(main_path, join(dict,file)))


    transformList = []
    normalize = transforms.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252])
    transformList.append(transforms.Resize([576,576]))
    transformList.append(transforms.Resize([448,448]))
    transformList.append(transforms.ToTensor())
    transformList.append(normalize)
    self.transformSequence = transforms.Compose(transformList)


  def __getitem__(self, index):
    imagePath = self.img_list[index]
    imageData = Image.open(imagePath).convert('RGB')
    imageData = self.transformSequence(imageData)
    return imageData, imagePath

  def __len__(self):

    return len(self.img_list)

dataset = Rot_dataset()
print(len(dataset))
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=10, pin_memory=True, shuffle=True)

model = resnet50(num_classes= 1)
checkpoint = torch.load(weight, map_location='cpu')
state_dict = {k.replace("module.", ""): v for k, v in checkpoint['model'].items()}
model.load_state_dict(state_dict)
print("Loading pretrianed weight from: {}".format(weight))
if torch.cuda.is_available():
    model = torch.nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())])
    model = model.cuda()
    cudnn.benchmark = True

with open("plco_rot.txt", "w") as fw:

    with torch.no_grad():
        for i, (images, imagePath) in enumerate(loader):
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)

            pred = torch.sigmoid(model(images))
            tp1 = pred.cpu()
            tp1 = tp1.tolist()
            for value,str in zip (tp1, imagePath):
                fw.write("{} {}\n".format(value[0],str))
                print("{} {}\n".format(value[0],str))

            fw.flush()
