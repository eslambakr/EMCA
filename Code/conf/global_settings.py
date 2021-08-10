import os
from datetime import datetime

#CIFAR100 dataset path (python version)
#CIFAR100_PATH = '/nfs/private/cifar100/cifar-100-python'

#mean and std of cifar100 dataset
CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

#CIFAR100_TEST_MEAN = (0.5088964127604166, 0.48739301317401956, 0.44194221124387256)
#CIFAR100_TEST_STD = (0.2682515741720801, 0.2573637364478126, 0.2770957707973042)

# data_set type
data_type = "tiny-imagenet"
if data_type == "cifar100":
    IMG_SIZE = 32
elif data_type == "tiny-imagenet":
    IMG_SIZE = 64
elif data_type == "dogs":
    IMG_SIZE = 128
elif data_type == "imagenet":
    IMG_SIZE = 224

#directory to save weights file
CHECKPOINT_PATH = 'checkpoint'

#total training epoches
EPOCH = 230 
MILESTONES = [60, 120, 160, 200]

#initial learning rate
#INIT_LR = 0.1

#time of we run the script
TIME_NOW = 'tiny_imagenet_self_local_ch_att_simple_3att'

#tensorboard log dir
LOG_DIR = 'runs'

#save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 100








