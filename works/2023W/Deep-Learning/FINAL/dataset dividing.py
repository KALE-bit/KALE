import os
import random
import shutil

data_dir = "E:/Repos/KALE/智能车视觉组/2023W/Deep-Learning/FINAL/已分类汽水图"
train_dir = "E:/Repos/KALE/智能车视觉组/2023W/Deep-Learning/FINAL/train"
test_dir = "E:/Repos/KALE/智能车视觉组/2023W/Deep-Learning/FINAL/test"
train_ratio = 0.7

if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

files = os.listdir(data_dir)
num_files = len(files)
num_train = int(num_files * train_ratio)
num_test = num_files - num_train

random.shuffle(files)
for i, file in enumerate(files):
    if i < num_train:
        dest_dir = train_dir
    else:
        dest_dir = test_dir
    shutil.copy(os.path.join(data_dir, file), os.path.join(dest_dir, file))
