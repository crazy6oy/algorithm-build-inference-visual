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

from wangyx.CalculateAP import ap_per_class

input_size = 380
TVT = "test"
data_dir = "/home/withai/wangyx/dataSetTemp/pictures"
classes = ["cvsiii,0", "cvsiii,1", "cvsiii,2", "cvsiii,-1"]
pathModel = f"{data_dir}/model/efficientnet-b4_CVSIII.pth"

pathTrue = f"txts/true_{data_dir}_{TVT}.txt"
pathFalse = f"txts/false_{data_dir}_{TVT}.txt"


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
            num_workers=8,
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
                ["0", "1", "2", "3"].index(
                    path_img.split("\\" if "win" in system else "/")[-2]
                )
            ).long(),
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
    outDict = {}
    dictRecognizeResult = {}

    dset_loaders, dset_sizes = loaddata(
        data_dir=data_dir, batch_size=2, set_name=TVT, shuffle=False
    )
    # fileTrue = open(pathTrue, "w")
    # fileFalse = open(pathFalse, "w")
    for data in tqdm.tqdm(dset_loaders[TVT]):
        inputs, labels, names = data
        labels = torch.squeeze(labels.type(torch.LongTensor))
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        outputs = model(inputs)

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

        for i, tf in enumerate(preds == labels):
            lab = labels.cpu().tolist()
            if tf:
                a = 1
            else:
                a = 0
            if lab[i] not in outDict.keys():
                outDict[lab[i]] = {}
            if a not in outDict[lab[i]].keys():
                outDict[lab[i]][a] = []
            outDict[lab[i]][a].append(names[i])

        outName.extend(names)

        # save recognize result
        for ind in range(len(names)):
            dictRecognizeResult[names[ind]] = [preds[ind], labels[ind]]

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        cont += 1
    # fileTrue.close()
    # fileFalse.close()
    print(outDict)
    # print(outDict[1][0])
    print("----")
    print(dictRecognizeResult)
    # for x in outDict[1][0]:
    #     print(x)
    print("----")
    # for x in list(set([x[:x.find("_")] for x in outDict[1][0]])):
    #     lenx = len([y for y in outDict[1][0] if y[:y.find("_")] == x])
    #     print(f"{x}: {lenx}")
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
        f"p {p.tolist()}\nr {r.tolist()}\nap {ap.tolist()}\nf1 {f1.tolist()}\nunique_classes:{unique_classes}\ncount:{count}"
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
    for key in dictConfusionMatrix.keys():
        if key[-1] == "0":
            print()
        print(dictConfusionMatrix[key], end=" ")
    # ----------------------------混淆矩阵----------------------------

    # ------------------------------细分------------------------------
    # 1.class 2.DatasetName 3.operationName 4.timeBucket 5.labelClass->classificationClass

    # 保存细分字典dictSubdivision
    if False:
        # 测试集的手术名称
        listOperationName = [
            "2020-01-15_1732_VID0001aa",
            "LC-CSR-16",
            "LC-CSR-38",
            "LC-CSR-82",
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


if __name__ == "__main__":
    input_size = 380
    TVT = "test"
    data_dir = "/public/ss/guanijnye/wangyx/dataset/LPDFrame1FPS"
    dict_classes = {"phase": ["0", "1"]}
    rule = "phase"
    classes = dict_classes[rule]
    class_num = len(classes)
    pathModel = f"{data_dir}/model/efficientnet-b4_CVSIII.pth"

    pathTrue = f"txts/true_{data_dir}_{TVT}.txt"
    pathFalse = f"txts/false_{data_dir}_{TVT}.txt"

    # test
    print("-" * 10)
    print("Test Accuracy:")
    model = torch.load(pathModel)

    print(f"加载模型:　{pathModel}")
    criterion = nn.CrossEntropyLoss().cuda()
    test_model(model, criterion)
