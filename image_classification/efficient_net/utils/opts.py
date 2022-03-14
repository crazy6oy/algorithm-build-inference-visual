# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import argparse

"""
use_gpu = torch.cuda.is_available()
batch_size = 2
lrInit = 0.001
momentum = 0.9
num_epochs = 128
input_size = 380
input_channels = 3
net_name = "efficientnet-b4"
data_dir = "/public/ss/wyx/dataset/CVSOriginImage"
json_path = "/public/ss/wyx/dataset/3 points system/divide.json"
ORGAN_MODEL_PATH = "/public/ss/wyx/dataset/3 points system/organ_segmentation.pth"
log_folder = "log"
checkpoint_folder = "checkpoints"
rule = "phase"
log_file_path = os.path.join(log_folder, "log.txt")
dict_classes = {"phase": ["0", "1"]}
classes = dict_classes[rule]
num_classes = len(classes)
"""

parser = argparse.ArgumentParser(description="PyTorch implementation of Efficient Net")
# 训练低级参数
parser.add_argument("--num-epoch", type=int, default=128, help="train number epoch")
parser.add_argument(
    "--batch-size", type=int, default=16, help="each input net images number"
)
parser.add_argument(
    "--input-size", type=int, default=380, help="model input data size(W,H)"
)
parser.add_argument(
    "--input-channels", type=int, default=9, help="model input data size(C)"
)
parser.add_argument("--lr-init", type=float, default=0.001, help="study ratio")
parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
parser.add_argument("--weight-decay", type=float, default=0.0004, help="weight decay")

# 训练中级参数
parser.add_argument(
    "--net-name",
    type=str,
    default="efficientnet-b4",
    help="choise model size(b0, b1, ... ,b7)",
)
parser.add_argument("--use-cuda", type=bool, default=True, help="is using cuda")
parser.add_argument(
    "--is-train", type=bool, default=True, help="True is train, False is test"
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="None is no load weight, else string is model weight path",
)

# 训练高级参数
parser.add_argument(
    "--ext-data-mode",
    type=int,
    default=1,
    help="choise ext-data input mode, if None, do nothing",
)
