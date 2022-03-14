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

# 如果使用上面的Git工程的话这样导入
# from efficientnet.model import EfficientNet
# 如果使用pip安装的Efficient的话这样导入
from efficientnet_pytorch import EfficientNet
from wangyx.CalculateAP import ap_per_class
from utils.opts import parser

from image_segmentation.deeplab_v3plus.demo import OrganSegmentation

EXT_SEGMENTATION_MODE = 1


class dataset(Dataset):
    def __init__(self, dataset_name, tras, args, **kwargs):
        super(dataset, self).__init__()
        """
        dir_img: 图像路径
        tras: 图像变换相关东西
        categories: 类别名称
        self.list_path_img: 这个根据具体训练数据来源设计吧，只要获取所有用来训练的图像路径就行
        为了方便图像数据来源的多样性，设计一个函数获取图像路径和标签
        """
        self.tras = tras
        self.input_size = args.input_size
        self.ext_data_mode = args.ext_data_mode
        self.getDataMsg(
            kwargs["image_folder"],
            kwargs["label_path"],
            kwargs["rule"],
            dataset_name,
            kwargs["categories"],
        )
        if self.ext_data_mode == EXT_SEGMENTATION_MODE:
            self.inference_module = OrganSegmentation(
                "/public/ss/wyx/project/go/deepLabv3Plus08/run/custom/deeplab-resnet/model_best.pth"
            )

    def __len__(self):
        return len(self.list_img_path)

    def __getitem__(self, index):
        path_img = self.list_img_path[index]
        label = self.list_label[index]
        img = cv2.imread(path_img)
        h, w, _ = img.shape
        img = Image.fromarray(img.astype("uint8")).convert("RGB")
        if h > w:
            padh = 0
            padw = h - w
        else:
            padh = w - h
            padw = 0
        img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
        img_efficient = img.resize((self.input_size, self.input_size), Image.BILINEAR)

        data_input = self.tras(img_efficient)
        if self.ext_data_mode == EXT_SEGMENTATION_MODE:
            probability_matrix = self.inference_module.inference(img)
            probability_matrix = torch.nn.functional.interpolate(
                probability_matrix,
                (data_input.shape[1], data_input.shape[2]),
                mode="bilinear",
            )[0]
            probability_matrix = probability_matrix.cpu()
            outputs = torch.cat((data_input, probability_matrix), 0)
        else:
            outputs = data_input

        return outputs, label

    def getDataMsg(self, image_folder, json_path, rule, dataset_name, categories):
        # 获取图像路径和标签
        print("format data making ...")
        self.list_label = []
        with open(json_path, encoding="utf-8") as f:
            label_msg = json.load(f)
        self.categories = label_msg["categories"][rule]
        label_data = label_msg["labels"]

        list_img_name = []
        for categry in self.categories:
            list_img_name.extend(label_data[rule][dataset_name][categry])
            self.list_label.extend(
                [
                    self.categories.index(categry),
                ]
                * len(label_data[rule][dataset_name][categry])
            )
        self.list_img_path = [os.path.join(image_folder, x) for x in list_img_name]


def loaddata(set_name, shuffle, args, **kwargs):
    data_transforms = {
        "train": transforms.Compose(
            [
                # transforms.Resize(input_size),
                # transforms.CenterCrop(input_size),
                # transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                transforms.Normalize(
                    mean=(
                        0.0,
                        0.0,
                        0.0,
                    ),
                    std=(1.0, 1.0, 1.0),
                ),
            ]
        ),
        "valid": transforms.Compose(
            [
                # transforms.Resize(input_size),
                # transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(
                        0.0,
                        0.0,
                        0.0,
                    ),
                    std=(1.0, 1.0, 1.0),
                ),
            ]
        ),
        "test": transforms.Compose(
            [
                # transforms.Resize(input_size),
                # transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(
                        0.0,
                        0.0,
                        0.0,
                    ),
                    std=(1.0, 1.0, 1.0),
                ),
            ]
        ),
    }

    # image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in [set_name]}
    # num_workers = 0 if CPU else num_workers = 1
    image_datasets = {
        x: dataset(set_name, data_transforms[x], args, **kwargs) for x in [set_name]
    }
    dataset_loaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x],
            batch_size=args.batch_size,
            shuffle=shuffle,
            drop_last=True,
        )
        for x in [set_name]
    }
    data_set_sizes = len(image_datasets[set_name])
    return dataset_loaders, data_set_sizes


def train_model(model_ft, criterion, log_file, args, **kwargs):
    optimizer = optim.SGD(
        (model_ft.parameters()),
        lr=args.lr_init,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    log_file.write("Begin!!!\n")
    log_file.write(f"{datetime.datetime.now()}\n")

    train_loss = []
    since = time.time()
    best_model_wts = model_ft.state_dict()
    best_ap = 0.0
    dset_loaders, dset_sizes = loaddata("train", True, args, **kwargs)
    for epoch in range(args.num_epoch):

        print("Data Size", dset_sizes)
        print("Epoch {}/{}".format(epoch, args.num_epoch * args.batch_size))

        optimizer = lrD(optimizer, args.lr_init, epoch + 1)
        print("-" * 32)

        running_loss = 0.0
        running_corrects = 0
        count = 0

        model_ft.train()
        for data in dset_loaders["train"]:
            inputs, labels = data

            if args.use_cuda:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            outputs = model_ft(inputs)
            loss = criterion(outputs, labels)
            conf, preds = torch.max(outputs.data, 1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            count += 1
            # print(f"loss:{loss.data}")
            if count % 30 == 0 or outputs.size()[0] < args.batch_size:
                print("Epoch:{} count:{} loss:{:.3f}".format(epoch, count, loss.item()))
                train_loss.append(loss.item())
                print(f"labels: {labels.tolist()}")
                print(f"preds:  {preds.tolist()}")
                log_file.write(
                    "Epoch:{} count:{} loss:{:.3f}\n".format(epoch, count, loss.item())
                )
                log_file.flush()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dset_sizes
        epoch_acc = running_corrects.double() / dset_sizes

        if True:
            epoch_loss_v, epoch_acc_v, epoch_ap_v = valid_model(
                model_ft, criterion, "valid", args, **kwargs
            )

            print("#" * 32)
            print(
                "# Loss_train: {:.4f} Acc_train: {:.4f}".format(epoch_loss, epoch_acc)
            )
            print(
                "# Loss_valid: {:.4f} Acc_valid: {:.4f} ap_valid: {:.4f}".format(
                    epoch_loss_v, epoch_acc_v, epoch_ap_v
                )
            )
            print("#" * 32)
        log_file.write(
            ">> Loss_train: {:.4f} Acc_train: {:.4f}\n".format(epoch_loss, epoch_acc)
        )
        log_file.write(
            ">> Loss_valid: {:.4f} Acc_valid: {:.4f} ap_valid: {:.4f}\n\n".format(
                epoch_loss_v, epoch_acc_v, epoch_ap_v
            )
        )
        log_file.flush()

        os.makedirs(kwargs["checkpoint_folder"], exist_ok=True)
        model_out_path = os.path.join(
            kwargs["checkpoint_folder"], f"{args.net_name}_last.pth"
        )
        torch.save(model_ft, model_out_path)
        if epoch_ap_v > best_ap:
            best_ap = epoch_ap_v
            best_model_wts = model_ft.state_dict()
            model_out_path = os.path.join(
                kwargs["checkpoint_folder"], f"{args.net_name}_best_mAP.pth"
            )
            torch.save(model_ft, model_out_path)
        if best_ap > 0.999:
            break

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )

    return train_loss, best_model_wts


def valid_model(model, criterion, mode, args, **kwargs):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    cont = 0

    results = {"tp": [], "conf": [], "pre": [], "label": []}
    dset_loaders, dset_sizes = loaddata(mode, False, args, **kwargs)
    for data in tqdm.tqdm(dset_loaders[mode]):
        inputs, labels = data
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
        conf = nn.functional.softmax(outputs, 1)

        results["tp"].append((preds == labels).long().data.cpu())
        results["conf"].append(torch.max(conf, 1)[0].data.cpu())
        results["pre"].append(preds.data.cpu())
        results["label"].append(labels.data.cpu())

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    results["tp"] = torch.cat(results["tp"], 0)
    results["conf"] = torch.cat(results["conf"], 0)
    results["pre"] = torch.cat(results["pre"], 0)
    results["label"] = torch.cat(results["label"], 0)

    acc = torch.sum(results["pre"] == results["label"]) / results["label"].shape[0]
    p, r, ap, f1, unique_classes, count = ap_per_class(
        results["tp"].numpy(),
        results["conf"].numpy(),
        results["pre"].numpy(),
        results["label"].numpy(),
    )

    return running_loss / dset_sizes, acc, np.average(ap)


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


def run_efficient_net_train(image_folder, label_path, rule, save_folder):
    args = parser.parse_args()

    struct_time = time.localtime(time.time())
    now_time = f"{struct_time.tm_year}-{struct_time.tm_mon}-{struct_time.tm_mday}_{struct_time.tm_hour}:{struct_time.tm_min}:{struct_time.tm_sec}"
    if args.use_cuda:
        args.use_cuda = torch.cuda.is_available()

    checkpoint_folder = os.path.join(
        save_folder, "efficient_net", now_time, "checkpoints"
    )
    os.makedirs(checkpoint_folder, exist_ok=True)
    log_file_path = os.path.join(save_folder, "efficient_net", now_time, "train.log")
    os.makedirs(os.path.split(log_file_path)[0], exist_ok=True)

    with open(label_path, "r", encoding="utf-8") as f:
        labels_msg = json.load(f)
    categories = labels_msg["categories"][rule]
    num_classes = len(categories)

    model_ft = EfficientNet.from_pretrained(
        args.net_name, in_channels=args.input_channels, num_classes=num_classes
    )

    criterion = nn.CrossEntropyLoss()

    if args.use_cuda:
        model_ft = torch.nn.DataParallel(model_ft)
        model_ft = model_ft.cuda()
        criterion = criterion.cuda()

    if args.is_train:
        with open(log_file_path, "w") as log_file:
            log_file.flush()
            train_loss, best_model_wts = train_model(
                model_ft,
                criterion,
                log_file,
                args,
                image_folder=image_folder,
                label_path=label_path,
                rule=rule,
                categories=categories,
                checkpoint_folder=checkpoint_folder,
            )

    # test
    print("-" * 10)
    print("Test Results:")
    if args.checkpoint and os.path.exists(args.checkpoint):
        model_ft = torch.load(args.checkpoint)
        if args.use_cuda:
            model_ft = model_ft.cuda()
    else:
        model_ft.load_state_dict(best_model_wts)
    test_loss, test_acc, test_map = valid_model(
        model_ft,
        criterion,
        "test",
        args,
        image_folder=image_folder,
        label_path=label_path,
        rule=rule,
        categories=categories,
        checkpoint_folder=checkpoint_folder,
    )
    print(f"loss: {test_loss} acc: {test_acc} mAP: {test_map}")
