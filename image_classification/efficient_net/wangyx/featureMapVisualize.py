import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import tqdm
import os


def get_row_col(num_pic):
    squr = num_pic**0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    return row, col


def visualize_feature_map(img_batch, name):
    dirSave = f"featureMap2/{name.split('.')[0]}"
    os.makedirs(dirSave, exist_ok=True)
    img_batch = torch.squeeze(img_batch, 0)
    feature_map = img_batch.permute(1, 2, 0)
    feature_map = (
        (feature_map - feature_map.min().data)
        * 255.0
        / (feature_map.max() - feature_map.min())
    )
    feature_map = feature_map.detach().cpu().numpy().astype(np.uint8)
    # print(feature_map)
    # print(f"min: {feature_map.min()},max: {feature_map.max()}")
    feature_map_combination = []

    for j in tqdm.tqdm(range(feature_map.shape[2])):
        plt.figure()
        # feature_map = np.stack([feature_map[..., x] for x in range(0, feature_map.shape[2], 200)], 2)
        featureTemp = feature_map[..., j : (j + 1)]

        num_pic = featureTemp.shape[2]
        # row, col = get_row_col(num_pic)
        row = col = 1

        for i in range(0, num_pic):
            feature_map_split = featureTemp[:, :, i]
            feature_map_combination.append(feature_map_split)
            plt.subplot(row, col, i + 1)
            plt.imshow(feature_map_split)
            # axis('off')
            plt.axis("off")
            # title('feature_map_{}'.format(i))
            plt.title("{}".format(i + j))

        plt.savefig(f"{dirSave}/{name.split('.')[0]}_{j}.png")
        plt.close()

    # 各个特征图按1：1 叠加
    plt.figure()
    feature_map_sum = sum(ele for ele in feature_map_combination)
    plt.subplot(1, 1, 1)
    plt.imshow(feature_map_sum)
    plt.savefig(f"{dirSave}/{name.split('.')[0]}_sum.png")


def visualizeFeatureMap(img_batch, img):
    padNum = 5
    if len(img_batch.shape) == 3:
        img_batch = torch.squeeze(img_batch, 0)
        featureMap = img_batch.permute(1, 2, 0)

        if True:
            minDim = torch.min(featureMap.reshape((-1, featureMap.shape[-1])), 0)[0]
            maxDim = torch.max(featureMap.reshape((-1, featureMap.shape[-1])), 0)[0]
            featureMap = (featureMap - minDim) * 255.0 / (maxDim - minDim)
        else:
            featureMap = (
                (featureMap - featureMap.min().data)
                * 255.0
                / (featureMap.max() - featureMap.min())
            )
        featureMap = featureMap.detach().cpu().numpy().astype(np.uint8)
        featureMap = np.repeat(featureMap, 4, 0)
        featureMap = np.repeat(featureMap, 4, 1)

        featureMap = np.pad(
            featureMap,
            ((padNum, padNum), (padNum, padNum), (0, 0)),
            mode="constant",
            constant_values=255,
        )

        row, col = get_row_col(featureMap.shape[-1])
        featureMapSum = featureMap.copy()
        featureMap = np.pad(
            featureMap,
            ((0, 0), (0, 0), (0, row * col - featureMap.shape[-1])),
            mode="constant",
            constant_values=255,
        )
        featureMapOut = [
            np.concatenate([featureMap[..., x * col + y] for y in range(col)], 1)
            for x in range(row)
        ]

        featureMapOut = np.concatenate(featureMapOut, 0)
        cv2.imwrite("a.png", featureMapOut)

        featureMapSum = featureMapSum.sum(-1)
        featureMapSumMax = featureMapSum.max()
        featureMapSum = featureMapSum / featureMapSumMax * 255
        cv2.imwrite("b.png", featureMapSum)

        weightImg = featureMapSum[padNum:-padNum, padNum:-padNum] / 255
        H, W, C = img.shape
        pad_s = (
            (((W - H) // 2, W - H - (W - H) // 2), (0, 0), (0, 0))
            if H < W
            else ((0, 0), ((H - W) // 2, H - W - (H - W) // 2), (0, 0))
        )
        img = np.pad(img, pad_s, mode="constant", constant_values=0)
        weightRate = img.shape[0] // weightImg.shape[0]
        print(weightRate)
        imgL = weightRate * weightImg.shape[0]
        weightRate = np.repeat(weightImg, weightRate, 0)
        weightRate = np.repeat(weightImg, weightRate, 1)
        img = img.resize((imgL, imgL))
        print(weightRate.shape)
        print(img.shape)

        return featureMapOut, featureMapSum, padNum
    else:
        pass


def drawBar(listInput, nameSave):
    plt.figure(figsize=(20, 10))
    plt.bar(list(range(len(listInput))), listInput)
    plt.title(nameSave)
    plt.savefig(f"{'/home/withai/wangyx/CVS/CVSDataSave/20210111'}/{nameSave}.png")
    plt.close()


if __name__ == "__main__":
    cvsi0 = [
        365,
        379,
        632,
        1729,
        1297,
        386,
        1205,
        1345,
        1278,
        1542,
        401,
        1250,
        1731,
        1284,
        1772,
        91,
        969,
        330,
        715,
        1012,
        794,
        1732,
        943,
        47,
        1591,
        385,
        1355,
        1239,
        133,
        456,
        795,
        859,
        46,
        1664,
        720,
        1349,
        1194,
        496,
        72,
        1229,
        396,
        126,
        432,
        1720,
        1218,
        1376,
        1068,
        1627,
        1244,
        1583,
    ]
    cvsi1 = [
        1227,
        558,
        1103,
        1348,
        1409,
        11,
        1624,
        16,
        36,
        850,
        1405,
        1360,
        1737,
        1652,
        970,
        676,
        44,
        615,
        654,
        1170,
        804,
        1236,
        9,
        1604,
        656,
        710,
        1267,
        154,
        723,
        1032,
        532,
        660,
        1358,
        824,
        1563,
        1475,
        64,
        1394,
        675,
        896,
        523,
        860,
        1272,
        1541,
        1081,
        708,
        620,
        85,
        70,
        1392,
    ]
    cvsi2 = [
        777,
        1486,
        530,
        728,
        569,
        305,
        789,
        1101,
        1516,
        664,
        776,
        250,
        61,
        1598,
        334,
        1537,
        638,
        1396,
        544,
        1483,
        1554,
        597,
        207,
        914,
        803,
        22,
        1222,
        426,
        1182,
        1209,
        476,
        192,
        1150,
        1686,
        129,
        69,
        269,
        1742,
        907,
        1714,
        1460,
        1634,
        1678,
        528,
        524,
        1617,
        1781,
        204,
        1325,
        1339,
    ]
