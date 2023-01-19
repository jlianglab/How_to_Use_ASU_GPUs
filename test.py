from os.path import isfile, join
from PIL import Image
import numpy as np
import random

def disparity_normalization(disp):  # disp is an array in uint8 data type
    _min = np.amin(disp)
    _max = np.amax(disp)
    disp_norm = (disp - _min) * 255.0 / (_max - _min)
    return np.uint8(disp_norm)


normal = []
abnormal = []

with open("plco_rot.txt", "r") as fr:
    line = fr.readline().strip()
    while line:
        prob = float(line.split(' ')[0])
        path = line.split(' ')[1]
        if prob > 0.2:
            abnormal.append(path)
        else:
            normal.append(path)
        line = fr.readline().strip()

print("Collected all cases. Normal: {}, abnormal: {}".format(len(normal), len(abnormal)))
with open("train.txt", "w") as fw:
    for n in normal:
        fw.write(n.split("/")[-2]+"/"+ n.split("/")[-1]+"\n")


    for abn in abnormal:
        im = Image.open(abn)
        im = im.rotate(-90, Image.NEAREST)
        fw.write("{}/rot_{}\n".format(abn.split("/")[-2],abn.split("/")[-1]))
        im.save("{}/rot_{}".format('/'.join(abn.split("/")[:-1]), abn.split("/")[-1]))
        print("{}/rot_{}".format(abn.split("/")[-2],abn.split("/")[-1]))




# normal_sample_files = random.sample(normal, 120)
# for i, file in enumerate(normal_sample_files):
#     im = Image.open(file)
#     im = np.array(im)
#     im = disparity_normalization(im)
#     im = Image.fromarray(im)
#     im.save("normal/{}.jpeg".format(file.split('/')[-1]))
#
# abnormal_sample_files = random.sample(abnormal, 120)
# for i, file in enumerate(abnormal_sample_files):
#     im = Image.open(file)
#     im = np.array(im)
#     im = disparity_normalization(im)
#     im = Image.fromarray(im)
#     im.save("abnormal/{}.jpeg".format(file.split('/')[-1]))