import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import os
from PIL import Image
import sys
import tqdm
import json

from wangyx.CalculateAP import ap_per_class

input_size = 380
TVT = "test"
data_dir = "CVSNewDatasetUpload/cvsi"
classes = ["cvsi,0", "cvsi,1", "cvsi,2"]
pathModel = f"{data_dir}/model/efficientnet-b4_1.pth"

pathTrue = f"txts/true_{data_dir}_{TVT}.txt"
pathFalse = f"txts/false_{data_dir}_{TVT}.txt"
pathFCRecord = "featureMap2/weights.json"
pathFCCalRecord = "featureMap2/calculate.json"


def savePictureSub(dictSub):
    from shutil import copy

    # 1.class 2.DatasetName 3.operationName 4.timeBucket 5.labelClass->classificationClass
    dir0 = data_dir
    listDP = dirorpath(dir0, dir0)
    dirSave = r"/home/withai/wangyx/CVS/CVSDataSave/20210111"
    for cls in dictSub.keys():
        for datasetName in dictSub[cls].keys():
            for operationName in dictSub[cls][datasetName].keys():
                for timeBucket in dictSub[cls][datasetName][operationName].keys():
                    for labCls in dictSub[cls][datasetName][operationName][
                        timeBucket
                    ].keys():
                        for name in dictSub[cls][datasetName][operationName][
                            timeBucket
                        ][labCls]:
                            path0 = [
                                x for x in listDP if name == x.split("/")[-1].strip()
                            ][0]
                            dirMk = os.path.join(
                                dirSave,
                                cls,
                                datasetName,
                                operationName,
                                timeBucket,
                                labCls,
                            )
                            os.makedirs(dirMk, exist_ok=True)
                            pathSave = os.path.join(
                                dirSave,
                                cls,
                                datasetName,
                                operationName,
                                timeBucket,
                                labCls,
                                name,
                            )
                            copy(path0, pathSave)


def statistic(dictSub):
    # 1.class 2.DatasetName 3.operationName 4.timeBucket 5.labelClass->classificationClass
    for cls in dictSub.keys():
        for datasetName in dictSub[cls].keys():
            for operationName in dictSub[cls][datasetName].keys():
                print(operationName)
                for timeBucket in sorted(
                    list(dictSub[cls][datasetName][operationName].keys())
                ):
                    countT = 0
                    countF = 0
                    for labCls in dictSub[cls][datasetName][operationName][
                        timeBucket
                    ].keys():
                        countHave = len(
                            dictSub[cls][datasetName][operationName][timeBucket][labCls]
                        )
                        if labCls.split("_")[0][-1] == labCls.split("_")[1][-1]:
                            countT += countHave
                        else:
                            countF += countHave
                    print(f"{timeBucket} >> T:{countT} F:{countF}")


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


def loaddata(data_dir, batch_size, set_name, shuffle):
    data_transforms = {
        "train": transforms.Compose(
            [
                # transforms.Resize(input_size),
                # transforms.CenterCrop(input_size),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "valid": transforms.Compose(
            [
                # transforms.Resize(input_size),
                # transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "test": transforms.Compose(
            [
                # transforms.Resize(input_size),
                # transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    # image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in [set_name]}
    # num_workers = 0 if CPU else num_workers = 1
    image_datasets = {
        x: dataset(os.path.join(data_dir, x), data_transforms[x]) for x in [set_name]
    }
    dataset_loaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x],
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=1,
            drop_last=True,
        )
        for x in [set_name]
    }
    data_set_sizes = len(image_datasets[set_name])
    return dataset_loaders, data_set_sizes


class dataset(Dataset):
    def __init__(self, dir_img, tras):
        super(dataset, self).__init__()
        self.dir_img = dir_img
        self.tras = tras
        self.list_path_img = dirorpath(dir_img, dir_img)

    def __len__(self):
        return len(self.list_path_img)

    def __getitem__(self, index):
        path_img = self.list_path_img[index]
        # path_img = os.path.join(self.dir_img, path_c)
        img = cv2.imread(path_img)
        H, W, C = img.shape
        # img = np.pad(img, (((W - H) // 2, W - H - (W - H) // 2), (0, 0), (0, 0))) if H < W else np.pad(img, (
        #     (0, 0), ((H - W) // 2, H - W - (H - W) // 2), (0, 0)))
        img = Image.fromarray(img.astype("uint8")).convert("RGB")

        system = sys.platform
        out0 = self.tras(img)
        pad_s = (
            ((H - W) // 2, H - W - (H - W) // 2, 0, 0)
            if H > W
            else (0, 0, (W - H) // 2, W - H - (W - H) // 2)
        )
        out1 = nn.functional.pad(out0, pad_s, value=0)
        out2 = torch.nn.functional.interpolate(
            out1.unsqueeze(0), size=input_size, mode="area"
        )[0]

        return (
            out2,
            torch.tensor(
                classes.index(path_img.split("\\" if "win" in system else "/")[-2])
            ),
            path_img.split("/")[-1],
        )


def test_model(model, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    cont = 0
    outPre = []
    outLabel = []
    outConf = []
    outTp = []
    outName = []
    outCal = {}

    dset_loaders, dset_sizes = loaddata(
        data_dir=data_dir, batch_size=2, set_name=TVT, shuffle=False
    )
    # fileTrue = open(pathTrue, "w")
    # fileFalse = open(pathFalse, "w")
    for data in tqdm.tqdm(dset_loaders[TVT]):
        inputs, labels, names = data
        labels = torch.squeeze(labels.type(torch.LongTensor))
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        outputs, x, featureMap = model(
            inputs
        )  # , names, dirorpath(data_dir, data_dir))

        # 画全连接和输入向量计算结果
        # weightValue = model.state_dict()["_fc.weight"]
        # from wangyx.featureMapVisualize import drawBar
        # drawBar((x[0, :, 0, 0] * weightValue[0]).cpu().detach().tolist(), names[0].split(".")[0] + "CVSI0")
        # drawBar((x[0, :, 0, 0] * weightValue[1]).cpu().detach().tolist(), names[0].split(".")[0] + "CVSI1")
        # drawBar((x[0, :, 0, 0] * weightValue[2]).cpu().detach().tolist(), names[0].split(".")[0] + "CVSI2")
        # drawBar((x[1, :, 0, 0] * weightValue[0]).cpu().detach().tolist(), names[1].split(".")[0] + "CVSI0")
        # drawBar((x[1, :, 0, 0] * weightValue[1]).cpu().detach().tolist(), names[1].split(".")[0] + "CVSI1")
        # drawBar((x[1, :, 0, 0] * weightValue[2]).cpu().detach().tolist(), names[1].split(".")[0] + "CVSI2")

        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
        conff = nn.functional.softmax(outputs, 1)
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
        outName.extend(names)
        # for i, x in enumerate(preds == labels):
        #     if x:
        #         fileTrue.write(names[i] + "\n")
        #     else:
        #         fileFalse.write(names[i] + "\n")
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        cont += 1

        outCal[names[0].split(".")[0]] = {}
        outCal[names[0].split(".")[0]]["recognition"] = ["CVSI0", "CVSI1", "CVSI2"][
            preds.cpu().tolist()[0]
        ]
        outCal[names[0].split(".")[0]]["label"] = ["CVSI0", "CVSI1", "CVSI2"][
            labels.cpu().tolist()[0]
        ]
        outCal[names[0].split(".")[0]]["CVSI0"] = (
            (x[0, :, 0, 0] * weightCal[0]).cpu().detach().tolist()
        )
        outCal[names[0].split(".")[0]]["CVSI1"] = (
            (x[0, :, 0, 0] * weightCal[1]).cpu().detach().tolist()
        )
        outCal[names[0].split(".")[0]]["CVSI2"] = (
            (x[0, :, 0, 0] * weightCal[2]).cpu().detach().tolist()
        )
        outCal[names[0].split(".")[0]]["featureMap"] = featureMap[0].cpu().tolist()

        outCal[names[1].split(".")[0]] = {}
        outCal[names[1].split(".")[0]]["recognition"] = ["CVSI0", "CVSI1", "CVSI2"][
            preds.cpu().tolist()[1]
        ]
        outCal[names[1].split(".")[0]]["label"] = ["CVSI0", "CVSI1", "CVSI2"][
            labels.cpu().tolist()[1]
        ]
        outCal[names[1].split(".")[0]]["CVSI0"] = (
            (x[1, :, 0, 0] * weightCal[0]).cpu().detach().tolist()
        )
        outCal[names[1].split(".")[0]]["CVSI1"] = (
            (x[1, :, 0, 0] * weightCal[1]).cpu().detach().tolist()
        )
        outCal[names[1].split(".")[0]]["CVSI2"] = (
            (x[1, :, 0, 0] * weightCal[2]).cpu().detach().tolist()
        )
        outCal[names[1].split(".")[0]]["featureMap"] = featureMap[1].cpu().tolist()

    # fileTrue.close()
    # fileFalse.close()
    f = open(pathFCCalRecord, "w", encoding="utf-8")
    json.dump(outCal, f, indent=2, ensure_ascii=False)
    f.close()
    f.close()
    print("#" * 32)
    print(
        "# Loss_test: {:.4f} Acc_test: {:.4f}".format(
            running_loss / dset_sizes, running_corrects.double() / dset_sizes
        )
    )
    print("#" * 32)
    p, r, ap, f1, unique_classes, count = ap_per_class(
        outTp.numpy(), outConf.numpy(), outPre.numpy(), outLabel.numpy()
    )
    print(
        f"p:{p}, r:{r}, ap:{ap}, f1:{f1}, unique_classes:{unique_classes}, count:{count}"
    )

    # ----------------------------混淆矩阵----------------------------
    dictConfusionMatrix = {}
    for labelTemp in classes:
        for classificationTemp in classes:
            keyTemp = f"lable:{labelTemp}->classification:{classificationTemp}"
            if keyTemp not in dictConfusionMatrix:
                dictConfusionMatrix[keyTemp] = 0
    for i in range(len(outLabel)):
        keyTemp = f"lable:{classes[outLabel[i]]}->classification:{classes[outPre[i]]}"
        dictConfusionMatrix[keyTemp] += 1
    for key in dictConfusionMatrix.keys():
        print(f"{key}:{dictConfusionMatrix[key]}")
    # ----------------------------混淆矩阵----------------------------

    # ------------------------------细分------------------------------
    # 1.class 2.DatasetName 3.operationName 4.timeBucket 5.labelClass->classificationClass

    # 保存细分字典dictSubdivision
    if False:
        # 测试集的手术名称
        # listOperationName = ["LC-CSR-4", "LC-CSR-9", "LC-CSR-31", "LC-CSR-87", "TBL-5"]
        # 训练集的手术名称
        listOperationName = [
            "2020-01-15_1732_VID0001aa",
            "LC-CSR-63",
            "LC-CSR_8",
            "LC-CSR-013",
            "LC-CSR-100",
            "LC-CSR-11",
            "LC-CSR-12",
            "LC-CSR-16",
            "LC-CSR-17",
            "LC-CSR-18",
            "LC-CSR-19",
            "LC-CSR-20",
            "LC-CSR-33",
            "LC-CSR-35",
            "LC-CSR-39",
            "LC-CSR-52",
            "LC-CSR-53",
            "LC-CSR-57",
            "LC-CSR-6",
            "LC-CSR-64",
            "LC-CSR-80",
            "LC-CSR-81",
            "LC-CSR-82",
            "LC-CSR-83",
            "LC-CSR-84",
            "LC-CSR-86",
            "LC-CSR-90",
            "LC-PUB-21",
            "LC-PUB-59",
            "LC-PUB-65",
            "LC-PUB-68",
            "LC-PUB-70",
            "LC-PUB-78",
            "LC-unknow",
            "TBL-4",
        ]
        dictSubdivision = {}
        if data_dir not in dictSubdivision.keys():
            dictSubdivision[data_dir] = {}
        if TVT not in dictSubdivision[data_dir].keys():
            dictSubdivision[data_dir][TVT] = {}
        for operationName in listOperationName:
            if operationName not in dictSubdivision[data_dir][TVT].keys():
                dictSubdivision[data_dir][TVT][operationName] = {}
        for i, pictureName in enumerate(outName):
            operationName = [x for x in listOperationName if x in pictureName][0]

            timeBucket = pictureName.replace(operationName, "").split(".")[0]
            if "_" in timeBucket:
                timeBucket = timeBucket.split("_")[-1]
            elif "-" in timeBucket:
                timeBucket = timeBucket.split("-")[-1]
            timeBucket = f"{int(timeBucket) // 100}00"
            keyTemp = f"label{outLabel[i]}_classification{outPre[i]}"
            if timeBucket not in dictSubdivision[data_dir][TVT][operationName].keys():
                dictSubdivision[data_dir][TVT][operationName][timeBucket] = {}
            if (
                keyTemp
                not in dictSubdivision[data_dir][TVT][operationName][timeBucket].keys()
            ):
                dictSubdivision[data_dir][TVT][operationName][timeBucket][keyTemp] = []
            dictSubdivision[data_dir][TVT][operationName][timeBucket][keyTemp].append(
                pictureName
            )

        # 按照细分字典的格式保存图片
        savePictureSub(dictSubdivision)
        # 统计每个手术每100秒正确分类和错误分类的数量，和上一个保存的图片结合观察
        statistic(dictSubdivision)
    # ------------------------------细分------------------------------


# test
print("-" * 10)
print("Test Accuracy:")
model = torch.load(pathModel)

# for name in model.state_dict():
#     print(name)
#     print(model.state_dict()[name])
weightCal = model.state_dict()["_fc.weight"]
weightValue = model.state_dict()["_fc.weight"].cpu().tolist()
biasValue = model.state_dict()["_fc.bias"].cpu().tolist()
dictWeight = {
    "FC-weight-CVSI0": weightValue[0],
    "FC-weight-CVSI1": weightValue[1],
    "FC-weight-CVSI2": weightValue[2],
    "FC-bias-CVSI0": biasValue[0],
    "FC-bias-CVSI1": biasValue[1],
    "FC-bias-CVSI2": biasValue[2],
}
f = open(pathFCRecord, "w", encoding="utf-8")
json.dump(dictWeight, f, indent=2, ensure_ascii=False)
f.close()
# print(weightValue[0].sort(descending=True)[1].tolist()[:64])
# print(weightValue[1].sort(descending=True)[1].tolist()[:64])
# print(weightValue[2].sort(descending=True)[1].tolist()[:64])
# from wangyx.featureMapVisualize import drawBar
#
# drawBar(model.state_dict()['_fc.weight'][2].sort(descending=True)[0].tolist())

print(f"加载模型:　{pathModel}")
criterion = nn.CrossEntropyLoss().cuda()
test_model(model, criterion)

# 0T '2020-01-15_1732_VID0001aa_157.jpg', 'LC-CSR-16-433.jpg', 'LC-CSR-82-390.jpg'
# 0F '182.jpg', 'LC-CSR-16-422.jpg', '2020-01-15_1732_VID0001aa_205.jpg'
# 1T '237.jpg', '2020-01-15_1732_VID0001aa_250.jpg', 'LC-CSR-82-565.jpg'
# 1F 'LC-CSR-82-604.jpg', '2020-01-15_1732_VID0001aa_263.jpg', 'LC-CSR-38-524.jpg'
# 2T '2020-01-15_1732_VID0001aa_270.jpg', '2020-01-15_1732_VID0001aa_271.jpg', '2020-01-15_1732_VID0001aa_275.jpg'
# 2F 'LC-CSR-16-488.jpg', 'LC-CSR-38-621.jpg', '269.jpg'
