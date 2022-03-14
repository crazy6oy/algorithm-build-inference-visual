import os
import json
import cv2
import tqdm
import base64
import shutil
import random
import numpy as np

from utils.cfg import TRAIN_CATEGORY, COLORS, DIVIDE


def traversalDir(dir1, returnX="path"):
    """
    params:
        returnX: choise [path,name]
    """
    out = []
    list_name = os.listdir(dir1)
    for name in list_name:
        dp = os.path.join(dir1, name)
        if os.path.isfile(dp):
            # list_x = dp.replace(dir_local + separatorSymbol, "").split(separatorSymbol)
            # out.append(os.path.join(*list_x))
            if returnX == "path":
                out.append(dp)
            elif returnX == "name":
                out.append(name)
            else:
                raise ValueError("returnX choise in [path, name]")
        elif os.path.isdir(dp):
            out.extend(traversalDir(dp, returnX=returnX))
        else:
            print(f"不知道这是个啥{dp}")
    return out


def str2Image(str):
    """
    return:
        img BGR
    """
    image_str = str.encode("ascii")
    image_byte = base64.b64decode(image_str)

    img = cv2.imdecode(
        np.asarray(bytearray(image_byte), dtype="uint8"), cv2.IMREAD_COLOR
    )  # for jpg
    return img


# def statistic(origin_folder):
#     listJson = [x for x in os.listdir(origin_folder) if ".json" in x]
#
#     dict_record = {}
#     for json_name in tqdm.tqdm(listJson):
#         json_path = os.path.join(origin_folder, json_name)
#         with open(json_path, encoding="utf-8") as f:
#             label_msg = json.load(f)
#             f.close()
#         for set_msg in label_msg["shapes"]:
#             if set_msg["shape_type"] != "polygon":
#                 continue
#             label = set_msg["label"]
#             if label not in dict_record.keys():
#                 dict_record[label] = 0
#             dict_record[label] += 1
#
#     for x in sorted(dict_record.keys()):
#         print("{: <30}{}".format(x, dict_record[x]))


def mk_mask(origin_folder, save_folder):
    list_json = [x for x in os.listdir(origin_folder) if ".json" in x]
    origin_save_folder = os.path.join(save_folder, "image")
    mask_save_folder = os.path.join(save_folder, "mask")
    os.makedirs(origin_save_folder, exist_ok=True)
    os.makedirs(mask_save_folder, exist_ok=True)

    for json_name in tqdm.tqdm(list_json):
        json_path = os.path.join(origin_folder, json_name)
        with open(json_path, encoding="utf-8") as f:
            label_msg = json.load(f)
            f.close()
        img = str2Image(label_msg["imageData"])
        H, W, _ = img.shape
        mask = np.zeros_like(img, dtype=np.uint8)
        for set_msg in label_msg["shapes"]:
            if set_msg["shape_type"] != "polygon":
                continue
            label = set_msg["label"].lower()
            if label not in TRAIN_CATEGORY.keys():
                continue
            label_id = TRAIN_CATEGORY[label]
            pts = np.abs(set_msg["points"]).astype(np.int)
            cv2.fillPoly(mask, [pts], (label_id, label_id, label_id))

        if np.sum(mask) == 0:
            continue

        # ignore抹黑
        for set_msg in label_msg["shapes"]:
            if set_msg["label"] == "ignore" and set_msg["shape_type"] != "polygon":
                label_id = 0
                pts = np.abs(set_msg["points"]).astype(np.int)
                cv2.fillPoly(mask, [pts], (label_id, label_id, label_id))
                cv2.fillPoly(img, [pts], (label_id, label_id, label_id))

        cv2.imwrite(
            os.path.join(origin_save_folder, json_name.replace(".json", ".jpg")), img
        )
        cv2.imwrite(
            os.path.join(mask_save_folder, json_name.replace(".json", ".png")), mask
        )


def MkDivideOperation(origin_folder, save_folder):
    imgFolder = os.path.join(origin_folder, "origin")
    maskFolder = os.path.join(origin_folder, "mask")

    list_img = os.listdir(imgFolder)

    list_operation = []
    dict_dataset = {}
    for img_name in tqdm.tqdm(list_img):
        if "_" in img_name:
            operation_name = img_name[: img_name.rfind("_")]
            if "VID" in operation_name.split("_")[-1]:
                operation_name = operation_name[: operation_name.rfind("_")]
        elif "-" in img_name:
            operation_name = img_name[: img_name.rfind("-")]
            if len(operation_name.split("-")) == 4:
                operation_name = operation_name[: operation_name.rfind("-")]
        else:
            operation_name = "no name"
        if operation_name not in list_operation:
            list_operation.append(operation_name)
        dataset_name = [x for x in DIVIDE.keys() if operation_name in DIVIDE[x]][0]
        if dataset_name not in dict_dataset.keys():
            dict_dataset[dataset_name] = []
        dict_dataset[dataset_name].append(img_name)
        img_save_path = os.path.join(save_folder, dataset_name, "origin")
        mask_save_path = os.path.join(save_folder, dataset_name, "mask")
        os.makedirs(img_save_path, exist_ok=True)
        os.makedirs(mask_save_path, exist_ok=True)
        shutil.copy(
            os.path.join(imgFolder, img_name), os.path.join(img_save_path, img_name)
        )
        shutil.copy(
            os.path.join(maskFolder, img_name.replace(".jpg", ".png")),
            os.path.join(mask_save_path, img_name.replace(".jpg", ".png")),
        )

    divide_path = os.path.join(save_folder, "divide.json")
    with open(divide_path, "w", encoding="utf-8") as f:
        json.dump(dict_dataset, f, ensure_ascii=False, indent=2)
        f.close()
    for x in list_operation:
        print(x)


def addMask(mask_folder, img_folder, save_folder):
    colors = np.array(COLORS, dtype=np.uint)

    list_img_name = os.listdir(mask_folder)
    for img_name in tqdm.tqdm(list_img_name):
        img_path = os.path.join(img_folder, img_name.replace(".png", ".jpg"))
        mask_path = os.path.join(mask_folder, img_name)

        img = cv2.imread(img_path)
        mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2GRAY)

        H, W, _ = img.shape
        if H > W:
            img_center = img[round((H - W) / 2) : round((H - W) / 2) + W]
        elif H < W:
            img_center = img[:, round((W - H) / 2) : round((W - H) / 2) + H]
        else:
            img_center = img[:]
        # img_center = img[:]
        h, w, _ = img_center.shape

        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask = colors[mask.reshape(-1)].reshape(h, w, 3).astype(np.uint8)

        img_show = cv2.addWeighted(img_center, 0.6, mask, 0.4, 1)
        cv2.imwrite(os.path.join(save_folder, img_name), img_show)
        # cv2.imshow('a', img_show)
        # cv2.waitKey()


def val01(origin_folder):
    # 验证shape_type有多少种
    list_json = [x for x in os.listdir(origin_folder) if ".json" in x]

    list_record = []
    for json_name in tqdm.tqdm(list_json):
        json_path = os.path.join(origin_folder, json_name)
        with open(json_path, encoding="utf-8") as f:
            label_msg = json.load(f)
            f.close()
        for set_msg in label_msg["shapes"]:
            if set_msg["shape_type"] not in list_record:
                list_record.append(set_msg["shape_type"])
    print(list_record)


def LabelNameLower(origin_folder, save_folder):
    list_json = [x for x in os.listdir(origin_folder) if ".json" in x]

    listx = []
    for json_name in tqdm.tqdm(list_json):
        json_path = os.path.join(origin_folder, json_name)
        save_path = os.path.join(save_folder, json_name)
        with open(json_path, encoding="utf-8") as f:
            label_msg = json.load(f)
            f.close()

        for i, set_msg in enumerate(label_msg["shapes"]):
            set_msg["label"] = set_msg["label"].lower()
            label_msg["shapes"][i] = set_msg
            if set_msg["label"] not in listx:
                listx.append(set_msg["label"])
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(label_msg, f, ensure_ascii=False, indent=2)
            f.close()
    print(sorted(listx))


def LabelNameChange(origin_folder, save_folder):
    change = {
        "dissected windows in the \rhepatocystic triangle": "dissected windows in the hepatocystic triangle"
    }
    list_json = [x for x in os.listdir(origin_folder) if ".json" in x]

    listx = []
    for json_name in tqdm.tqdm(list_json):
        json_path = os.path.join(origin_folder, json_name)
        save_path = os.path.join(save_folder, json_name)
        with open(json_path, encoding="utf-8") as f:
            label_msg = json.load(f)
            f.close()

        for i, set_msg in enumerate(label_msg["shapes"]):
            set_msg["label"] = set_msg["label"].lower()
            if set_msg["label"] in change.keys():
                set_msg["label"] = change[set_msg["label"]]
            label_msg["shapes"][i] = set_msg
            if set_msg["label"] not in listx:
                listx.append(set_msg["label"])
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(label_msg, f, ensure_ascii=False, indent=2)
            f.close()
    print(sorted(listx))


def statistic(json_labeled_folder, list_json):
    dict_record = {}
    for json_name in tqdm.tqdm(list_json):
        json_path = os.path.join(json_labeled_folder, json_name)
        with open(json_path, encoding="utf-8") as f:
            label_msg = json.load(f)
            f.close()
        for set_msg in label_msg["shapes"]:
            if set_msg["shape_type"] != "polygon":
                continue
            label = set_msg["label"]
            if label not in TRAIN_CATEGORY:
                continue
            if label not in dict_record.keys():
                dict_record[label] = 0
            dict_record[label] += 1

    for x in sorted(dict_record.keys()):
        print("{}\t{}".format(x, dict_record[x]))


def DivideSurgery(json_labeled_folder):
    list_json_name = os.listdir(json_labeled_folder)

    dict_surgery_frame = {}
    for json_name in list_json_name:
        surgery_name = json_name.split("_")[0]
        if surgery_name not in dict_surgery_frame.keys():
            dict_surgery_frame[surgery_name] = []
        dict_surgery_frame[surgery_name].append(json_name)

    dict_surgery_divide_result = {"train": [], "valid": [], "test": []}
    while len(dict_surgery_frame.keys()) > 0:
        surgery_name = random.choice(list(dict_surgery_frame.keys()))

        if len(dict_surgery_divide_result["test"]) * 8 < len(
            dict_surgery_divide_result["train"]
        ):
            dataset_name = "test"
        elif len(dict_surgery_divide_result["valid"]) * 8 < len(
            dict_surgery_divide_result["train"]
        ):
            dataset_name = "valid"
        else:
            dataset_name = "train"

        if len(dict_surgery_frame[surgery_name]) > 70:
            dataset_name = "train"

        dict_surgery_divide_result[dataset_name].extend(
            dict_surgery_frame.pop(surgery_name)
        )
    for x in ["train", "valid", "test"]:
        print(x)
        print(list(set([y.split("_")[0] for y in dict_surgery_divide_result[x]])))
        statistic(json_labeled_folder, dict_surgery_divide_result[x])


if __name__ == "__main__":
    # val01(r"D:\work\dataSet\organSegmentation\origin\v0")
    # statistic(r"D:\work\dataSet\organSegmentation\processed",
    #           os.listdir(r"D:\work\dataSet\organSegmentation\processed"))

    # 标签名称全小写
    # origin_folder = r"D:\work\dataSet\organSegmentation\origin"
    # save_folder = r"D:\work\dataSet\organSegmentation\processed"
    # LabelNameLower(origin_folder, save_folder)

    # 修改标签（大小写+名称修改）
    # origin_folder = r"D:\work\dataSet\organSegmentation\origin"
    # save_folder = r"D:\work\dataSet\organSegmentation\processed"
    # LabelNameChange(origin_folder, save_folder)

    # 根据labelme标注文件生成标签
    origin_folder = r"D:\work\dataSet\organSegmentation\processed"
    save_folder = r"D:\work\dataSet\organSegmentation\image_mask"
    mk_mask(origin_folder, save_folder)

    # 划分训练验证和测试集
    # json_labeled_folder = r"D:\work\dataSet\organSegmentation\processed"
    # DivideSurgery(json_labeled_folder)

    # 生成训练验证和测试集
    # origin_folder = r"D:\work\dataSet\organSegmentation\origin_mask\data"
    # save_folder = r"D:\work\dataSet\organSegmentation\v3"
    # MkDivideOperation(origin_folder, save_folder)

    # 可视化模型输出结果
    # mask_folder = r"D:\work\dataSet\organSegmentation\image_mask\mask"
    # save_folder = r"D:\work\dataSet\organSegmentation\visual"
    # img_folder = r"D:\work\dataSet\organSegmentation\image_mask\image"
    # addMask(mask_folder, img_folder, save_folder)
