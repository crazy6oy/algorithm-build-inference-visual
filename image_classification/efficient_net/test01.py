from __future__ import print_function, division

import numpy as np
import tqdm
import json
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from torch.utils.data.dataset import Dataset
import time
import os
from PIL import Image, ImageOps
import sys
import datetime

from efficientnet_pytorch import EfficientNet
from wangyx.CalculateAP import ap_per_class
from utils.segmentation_inference import OrganSegmentation
from efficientnet_pytorch.utils import *
from efficientnet_pytorch.model import *


# 如果使用上面的Git工程的话这样导入
# from efficientnet.model import EfficientNet
# 如果使用pip安装的Efficient的话这样导入


def img_pre_process(img_src):
    img_src = cv2.resize(img_src, (630, 380))

    # 2.灰度处理与二值化
    img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.equalizeHist(img_gray)
    img_gray = 255 - img_gray
    ret, img_bin = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

    # 3.连通域分析
    # img_contour, contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(
        img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # 4.制作掩膜图片
    img_mask = np.zeros(img_src.shape, np.uint8)
    cv2.drawContours(img_mask, contours, -1, (255, 255, 255), -1)

    # ---------------------------------------------------------
    # cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(2, (5, 13))  # 定义结构元素的形状和大小
    img_mask = cv2.erode(img_mask, kernel)  # 腐蚀操作
    kernel = cv2.getStructuringElement(2, (5, 13))  # 定义结构元素的形状和大小
    img_mask = cv2.dilate(img_mask, kernel)  # 膨胀操作
    # ---------------------------------------------------------

    # 5.执行与运算
    img_result = cv2.bitwise_and(img_src, img_mask)

    return img_result


# some parameters
batch_size = 16
lrInit = 0.001
momentum = 0.9
num_epochs = 128
input_size = 380
net_name = "efficientnet-b4"
data_dir = r"D:\work\dataSet\public-dataset\happy-wd\test_images"
json_path = r"C:\Users\wangyx\Desktop\happy_WD\test.json"
log_folder = "./log"
os.makedirs(log_folder, exist_ok=True)
checkpoint_folder = "./checkpoints"
os.makedirs(checkpoint_folder, exist_ok=True)
rule = "wd"
use_gpu = True

dict_batch = {"train": 16, "valid": 4, "test": 2}
dict_classes = {
    "wd": [
        "frasiers_dolphin",
        "rough_toothed_dolphin",
        "pygmy_killer_whale",
        "commersons_dolphin",
        "globis",
        "pantropic_spotted_dolphin",
        "brydes_whale",
        "white_sided_dolphin",
        "long_finned_pilot_whale",
        "pilot_whale",
        "cuviers_beaked_whale",
        "common_dolphin",
        "short_finned_pilot_whale",
        "sei_whale",
        "spotted_dolphin",
        "southern_right_whale",
        "kiler_whale",
        "bottlenose_dolpin",
        "gray_whale",
        "fin_whale",
        "killer_whale",
        "minke_whale",
        "melon_headed_whale",
        "spinner_dolphin",
        "dusky_dolphin",
        "false_killer_whale",
        "blue_whale",
        "humpback_whale",
        "beluga",
        "bottlenose_dolphin",
    ]
}
classes = dict_classes[rule]
class_num = len(classes)


def dirorpath(dir1, dir_local):
    out = []
    list_name = os.listdir(dir1)
    for name in list_name:
        dp = os.path.join(dir1, name)
        if os.path.isfile(dp):
            list_x = dp.replace(dir_local + "\\", "").split("\\")
            out.append(os.path.join(*list_x))
        elif os.path.isdir(dp):
            out.extend(dirorpath(dp, dir_local))
        else:
            print(f"不知道这是个啥{dp}")
    return out


class dataset(Dataset):
    def __init__(self, image_folder, json_path, rule, dataset_name, categories, tras):
        super(dataset, self).__init__()
        """
        dir_img: 图像路径
        tras: 图像变换相关东西
        categories: 类别名称
        self.list_path_img: 这个根据具体训练数据来源设计吧，只要获取所有用来训练的图像路径就行
        为了方便图像数据来源的多样性，设计一个函数获取图像路径和标签
        """
        self.tras = tras
        self.dataset_name = dataset_name
        self.getDataMsg(image_folder, json_path, rule, dataset_name, categories)

    def __len__(self):
        return len(self.list_img_path)

    def __getitem__(self, index):
        path_img = self.list_img_path[index]
        label = self.list_label[index]
        img = cv2.imread(path_img)

        # ----------------------------------------------------
        # 图像预处理
        img = img_pre_process(img)
        # ----------------------------------------------------

        h, w, _ = img.shape
        img = Image.fromarray(img.astype("uint8")).convert("RGB")
        if h > w:
            padh = 0
            padw = h - w
        else:
            padh = w - h
            padw = 0
        img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
        img_efficient = img.resize((input_size, input_size), Image.BILINEAR)

        data_input = self.tras(img_efficient)

        if self.dataset_name != "test":
            return data_input, label
        else:
            return data_input, label, os.path.split(path_img)[-1]

    def getDataMsg(self, image_folder, json_path, rule, dataset_name, categories):
        # 获取图像路径和标签
        print("format data making ...")
        self.list_label = []
        with open(json_path, encoding="utf-8") as f:
            label_msg = json.load(f)
            f.close()
        self.categories = categories

        list_img_name = []
        for categry in self.categories:
            if categry not in label_msg[rule][dataset_name]:
                continue
            list_img_name.extend(label_msg[rule][dataset_name][categry])
            self.list_label.extend(
                [
                    self.categories.index(categry),
                ]
                * len(label_msg[rule][dataset_name][categry])
            )
        self.list_img_path = [os.path.join(image_folder, x) for x in list_img_name]


def loaddata(data_dir, json_path, rule, categories, batch_size, set_name, shuffle):
    data_transforms = {
        "train": transforms.Compose(
            [
                # transforms.Resize(input_size),
                # transforms.CenterCrop(input_size),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        ),
        "valid": transforms.Compose(
            [
                # transforms.Resize(input_size),
                # transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        ),
        "test": transforms.Compose(
            [
                # transforms.Resize(input_size),
                # transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        ),
    }

    # image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in [set_name]}
    # num_workers = 0 if CPU else num_workers = 1
    image_datasets = {
        x: dataset(data_dir, json_path, rule, set_name, categories, data_transforms[x])
        for x in [set_name]
    }
    dataset_loaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x],
            batch_size=dict_batch[x],
            shuffle=shuffle,
            num_workers=2,
            drop_last=True,
        )
        for x in [set_name]
    }
    data_set_sizes = len(image_datasets[set_name])
    return dataset_loaders, data_set_sizes


def train_model(model_ft, criterion, optimizer, lrInit, num_epochs=8):
    pathLog = os.path.join(log_folder, "log.txt")
    with open(pathLog, "a") as fileLog:
        fileLog.write("Begin！！！\n")
        fileLog.write(f"{datetime.datetime.now()}\n")
        print("日志写入开始！！！")
        fileLog.close()

    train_loss = []
    since = time.time()
    best_model_wts = model_ft.state_dict()
    best_acc = 0.0
    best_ap = 0.0
    dset_loaders, dset_sizes = loaddata(
        data_dir, json_path, rule, classes, batch_size, "train", True
    )
    for epoch in range(num_epochs):

        print("Data Size", dset_sizes)
        print("Epoch {}/{}".format(epoch, num_epochs - 1))

        optimizer = lrD(optimizer, lrInit, epoch + 1)
        print("-" * 32)

        running_loss = 0.0
        running_corrects = 0
        count = 0

        model_ft.train()

        for data in tqdm.tqdm(dset_loaders["train"], desc=f"epoch: {epoch}"):
            inputs, labels = data

            # -----------------------CELoss使用-----------------------
            # labels = torch.squeeze(labels.type(torch.LongTensor))
            # -----------------------CELoss使用-----------------------
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            outputs = model_ft(inputs)
            loss = criterion(outputs, labels)
            conf, preds = torch.max(outputs.data, 1)
            # _, labels = torch.max(labels.data, 1)
            # print(f"labels: {labels.tolist()}")
            # print(f"preds:  {preds.tolist()}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            count += 1
            # print(f"loss:{loss.data}")
            train_loss.append(loss.item())
            if count % 30 == 0 or outputs.size()[0] < batch_size:
                print("Epoch:{} count:{} loss:{:.3f}".format(epoch, count, loss.item()))
                print(f"labels: {labels.tolist()}")
                print(f"preds:  {preds.tolist()}")
                with open(pathLog, "a") as fileLog:
                    fileLog.write(
                        "Epoch:{} count:{} loss:{:.3f}\n".format(
                            epoch, count, loss.item()
                        )
                    )

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            if count > 2000:
                break

        epoch_loss = running_loss / dset_sizes
        epoch_acc = running_corrects.double() / dset_sizes

        # ----------------------万一验证显存不够----------------------
        model_out_path = f"./{net_name}_bak.pth"
        torch.save(model_ft, model_out_path)
        # ----------------------万一验证显存不够----------------------

        # ------------------------------验证------------------------------------
        epoch_loss_v, epoch_acc_v, epoch_ap_v = valid_model(model_ft, criterion)
        # ------------------------------验证------------------------------------

        print("#" * 32)
        print("# Loss_train: {:.4f} Acc_train: {:.4f}".format(epoch_loss, epoch_acc))
        print(
            "# Loss_valid: {:.4f} Acc_valid: {:.4f} ap_valid: {:.4f}".format(
                epoch_loss_v, epoch_acc_v, epoch_ap_v
            )
        )
        print("#" * 32)
        with open(pathLog, "a", encoding="utf-8") as fileLog:
            fileLog.write(
                ">> Loss_train: {:.4f} Acc_train: {:.4f}\n".format(
                    epoch_loss, epoch_acc
                )
            )
            fileLog.write(
                ">> Loss_valid: {:.4f} Acc_valid: {:.4f} ap_valid: {:.4f}\n\n".format(
                    epoch_loss_v, epoch_acc_v, epoch_ap_v
                )
            )
            fileLog.close()

        save_dir = checkpoint_folder
        model_out_path = f"./{net_name}_last.pth"
        torch.save(model_ft, model_out_path)
        if epoch_ap_v > best_ap:
            best_ap = epoch_ap_v
            best_model_wts = model_ft.state_dict()
            model_out_path = f"./{net_name}_best_mAP_{epoch}.pth"
            torch.save(model_ft, model_out_path)

        if best_ap > 0.999 and epoch_acc_v > 0.999 and epoch > 64:
            break

    # save best model
    save_dir = data_dir + "/model"
    os.makedirs(save_dir, exist_ok=True)
    model_ft.load_state_dict(best_model_wts)
    model_out_path = save_dir + "/" + net_name + "Best.pth"
    torch.save(model_ft, model_out_path)

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )

    return train_loss, best_model_wts


def valid_model(model, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    cont = 0
    dset_loaders, dset_sizes = loaddata(
        data_dir, json_path, rule, classes, batch_size, "valid", False
    )
    for data in tqdm.tqdm(dset_loaders["valid"]):
        inputs, labels = data
        # labels = torch.squeeze(labels.type(torch.LongTensor))
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
        conff = nn.functional.softmax(outputs, 1)
        # if cont == 0:
        #     outPre = outputs.data.cpu()
        #     outLabel = labels.data.cpu()
        # else:
        #     outPre = torch.cat((outPre, outputs.data.cpu()), 0)
        #     outLabel = torch.cat((outLabel, labels.data.cpu()), 0)
        if cont == 0:
            outPre = preds.data.cpu()
            outLabel = labels.data.cpu()
            outConf = torch.max(conff, 1)[0].data.cpu()
            outTp = (preds == labels).long().data.cpu()
        else:
            outPre = torch.cat((outPre, preds.data.cpu()), 0)
            outLabel = torch.cat((outLabel, labels.data.cpu()), 0)
            outConf = torch.cat((outConf, torch.max(conff, 1)[0].data.cpu()), 0)
            outTp = torch.cat((outTp, (preds == labels).long().data.cpu()), 0)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        cont += 1

    acc = 0
    # outPre = torch.argmax(outPre, 1)
    acc = torch.sum(outPre == outLabel) / outLabel.shape[0]
    p, r, ap, f1, unique_classes, count = ap_per_class(
        outTp.numpy(), outConf.numpy(), outPre.numpy(), outLabel.numpy()
    )

    return (
        running_loss / dset_sizes,
        acc,
        np.average(ap),
    )  # running_corrects.double() / dset_sizes


def test_model(model, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    cont = 0
    outPre = []
    outLabel = []
    dset_loaders, dset_sizes = loaddata(
        data_dir, json_path, rule, classes, batch_size, "test", False
    )
    for data in dset_loaders["test"]:
        inputs, labels, img_path = data
        labels = torch.squeeze(labels.type(torch.LongTensor))
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
        if cont == 0:
            outPre = outputs.data.cpu()
            outLabel = labels.data.cpu()
        else:
            outPre = torch.cat((outPre, outputs.data.cpu()), 0)
            outLabel = torch.cat((outLabel, labels.data.cpu()), 0)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        cont += 1

        for i in range(len(preds)):
            print(f"{img_path[i]} {dict_classes['wd'][preds[i]]}")
    print("#" * 32)
    print(
        "# Loss_test: {:.4f} Acc_test: {:.4f}".format(
            running_loss / dset_sizes, running_corrects.double() / dset_sizes
        )
    )
    print("#" * 32)


def exp_lr_scheduler(optimizer, epoch, init_lr=0.01, lr_decay_epoch=10):
    """Decay learning rate by a f#            model_out_path ="./model/W_epoch_{}.pth".format(epoch)
    #            torch.save(model_W, model_out_path) actor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.8 ** (epoch // lr_decay_epoch))
    print("LR is set to {}".format(lr))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return optimizer


def lrD(optimizer, lrInit, epochNum):
    lrMin = 0.0003
    ratioD = 0.8
    epochD = 20
    lr = lrInit
    tally = 0
    for i in range(epochNum // epochD):
        lr *= ratioD
        if lr < lrMin:
            tally += 1
            lr = ratioD**tally * lrInit
    for group in optimizer.param_groups:
        group["lr"] = lr
    return optimizer


if __name__ == "__main__":
    import shutil

    # train
    pth_map = {
        "efficientnet-b0": "efficientnet-b0-355c32eb.pth",
        "efficientnet-b1": "efficientnet-b1-f1951068.pth",
        "efficientnet-b2": "efficientnet-b2-8bb594d6.pth",
        "efficientnet-b3": "efficientnet-b3-5fb5a3c3.pth",
        "efficientnet-b4": "efficientnet-b4-6ed6700e.pth",
        "efficientnet-b5": "efficientnet-b5-b6417697.pth",
        "efficientnet-b6": "efficientnet-b6-c76e70fd.pth",
        "efficientnet-b7": "efficientnet-b7-dcc49843.pth",
    }
    # 自动下载到本地预训练
    # model = EfficientNet.from_pretrained(net_name)
    # shutil.copy("/root/.cache/torch/hub/checkpoints/efficientnet-b4-6ed6700e.pth", "./efficientnet-b4-6ed6700e.pth")
    # 离线加载预训练，需要事先下载好
    # model_ft = EfficientNet.from_name(net_name, in_channels=3)
    net_weight = r"C:\Users\wangyx\Desktop\happy_WD\efficientnet-b4_best_mAP_1.pth"
    model_ft = torch.load(net_weight)
    model_ft = model_ft.cuda()

    print("-" * 10)
    print("Test Accuracy:")
    criterion = nn.CrossEntropyLoss().cuda()
    test_model(model_ft, criterion)
