# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import torch.utils.data as data

from PIL import Image
import os
import cv2
import numpy as np
from numpy.random import randint
import torch
import random
import json
import tqdm
import time


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class TSNDataSet(data.Dataset):
    def __init__(
        self,
        root_path,
        list_file,
        num_segments=3,
        new_length=1,
        modality="RGB",
        image_tmpl="img_{:05d}.jpg",
        transform=None,
        random_shift=True,
        test_mode=False,
        remove_missing=False,
        dense_sample=False,
        twice_sample=False,
    ):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.remove_missing = remove_missing
        self.dense_sample = dense_sample  # using dense sample as I3D
        self.twice_sample = twice_sample  # twice sample for more validation
        if self.dense_sample:
            print("=> Using dense sample for the dataset...")
        if self.twice_sample:
            print("=> Using twice sample for the dataset...")

        if self.modality == "RGBDiff":
            self.new_length += 1  # Diff needs one more image to calculate diff

        self._parse_list()

    def _load_image(self, directory, idx):
        if self.modality == "RGB" or self.modality == "RGBDiff":
            try:
                return [
                    Image.open(
                        os.path.join(
                            self.root_path, directory, self.image_tmpl.format(idx)
                        )
                    ).convert("RGB")
                ]
            except Exception:
                print(
                    "error loading image:",
                    os.path.join(
                        self.root_path, directory, self.image_tmpl.format(idx)
                    ),
                )
                return [
                    Image.open(
                        os.path.join(
                            self.root_path, directory, self.image_tmpl.format(1)
                        )
                    ).convert("RGB")
                ]
        elif self.modality == "Flow":
            if self.image_tmpl == "flow_{}_{:05d}.jpg":  # ucf
                x_img = Image.open(
                    os.path.join(
                        self.root_path, directory, self.image_tmpl.format("x", idx)
                    )
                ).convert("L")
                y_img = Image.open(
                    os.path.join(
                        self.root_path, directory, self.image_tmpl.format("y", idx)
                    )
                ).convert("L")
            elif self.image_tmpl == "{:06d}-{}_{:05d}.jpg":  # something v1 flow
                x_img = Image.open(
                    os.path.join(
                        self.root_path,
                        "{:06d}".format(int(directory)),
                        self.image_tmpl.format(int(directory), "x", idx),
                    )
                ).convert("L")
                y_img = Image.open(
                    os.path.join(
                        self.root_path,
                        "{:06d}".format(int(directory)),
                        self.image_tmpl.format(int(directory), "y", idx),
                    )
                ).convert("L")
            else:
                try:
                    # idx_skip = 1 + (idx-1)*5
                    flow = Image.open(
                        os.path.join(
                            self.root_path, directory, self.image_tmpl.format(idx)
                        )
                    ).convert("RGB")
                except Exception:
                    print(
                        "error loading flow file:",
                        os.path.join(
                            self.root_path, directory, self.image_tmpl.format(idx)
                        ),
                    )
                    flow = Image.open(
                        os.path.join(
                            self.root_path, directory, self.image_tmpl.format(1)
                        )
                    ).convert("RGB")
                # the input flow file is RGB image with (flow_x, flow_y, blank) for each channel
                flow_x, flow_y, _ = flow.split()
                x_img = flow_x.convert("L")
                y_img = flow_y.convert("L")

            return [x_img, y_img]

    def _parse_list(self):
        # check the frame number is large >3:
        tmp = [x.strip().split(" ") for x in open(self.list_file)]
        if not self.test_mode or self.remove_missing:
            tmp = [item for item in tmp if int(item[1]) >= 3]
        self.video_list = [VideoRecord(item) for item in tmp]

        if self.image_tmpl == "{:06d}-{}_{:05d}.jpg":
            for v in self.video_list:
                v._data[1] = int(v._data[1]) / 2
        print("video number:%d" % (len(self.video_list)))

    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        """
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [
                (idx * t_stride + start_idx) % record.num_frames
                for idx in range(self.num_segments)
            ]
            return np.array(offsets) + 1
        else:  # normal sample
            average_duration = (
                record.num_frames - self.new_length + 1
            ) // self.num_segments
            if average_duration > 0:
                offsets = np.multiply(
                    list(range(self.num_segments)), average_duration
                ) + randint(average_duration, size=self.num_segments)
            elif record.num_frames > self.num_segments:
                offsets = np.sort(
                    randint(
                        record.num_frames - self.new_length + 1, size=self.num_segments
                    )
                )
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets + 1

    def _get_val_indices(self, record):
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [
                (idx * t_stride + start_idx) % record.num_frames
                for idx in range(self.num_segments)
            ]
            return np.array(offsets) + 1
        else:
            if record.num_frames > self.num_segments + self.new_length - 1:
                tick = (record.num_frames - self.new_length + 1) / float(
                    self.num_segments
                )
                offsets = np.array(
                    [int(tick / 2.0 + tick * x) for x in range(self.num_segments)]
                )
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets + 1

    def _get_test_indices(self, record):
        if self.dense_sample:
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_list = np.linspace(0, sample_pos - 1, num=10, dtype=int)
            offsets = []
            for start_idx in start_list.tolist():
                offsets += [
                    (idx * t_stride + start_idx) % record.num_frames
                    for idx in range(self.num_segments)
                ]
            return np.array(offsets) + 1
        elif self.twice_sample:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

            offsets = np.array(
                [int(tick / 2.0 + tick * x) for x in range(self.num_segments)]
                + [int(tick * x) for x in range(self.num_segments)]
            )

            return offsets + 1
        else:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array(
                [int(tick / 2.0 + tick * x) for x in range(self.num_segments)]
            )
            return offsets + 1

    def __getitem__(self, index):
        record = self.video_list[index]
        # check this is a legit video folder

        if self.image_tmpl == "flow_{}_{:05d}.jpg":
            file_name = self.image_tmpl.format("x", 1)
            full_path = os.path.join(self.root_path, record.path, file_name)
        elif self.image_tmpl == "{:06d}-{}_{:05d}.jpg":
            file_name = self.image_tmpl.format(int(record.path), "x", 1)
            full_path = os.path.join(
                self.root_path, "{:06d}".format(int(record.path)), file_name
            )
        else:
            file_name = self.image_tmpl.format(1)
            full_path = os.path.join(self.root_path, record.path, file_name)

        while not os.path.exists(full_path):
            print(
                "################## Not Found:",
                os.path.join(self.root_path, record.path, file_name),
            )
            index = np.random.randint(len(self.video_list))
            record = self.video_list[index]
            if self.image_tmpl == "flow_{}_{:05d}.jpg":
                file_name = self.image_tmpl.format("x", 1)
                full_path = os.path.join(self.root_path, record.path, file_name)
            elif self.image_tmpl == "{:06d}-{}_{:05d}.jpg":
                file_name = self.image_tmpl.format(int(record.path), "x", 1)
                full_path = os.path.join(
                    self.root_path, "{:06d}".format(int(record.path)), file_name
                )
            else:
                file_name = self.image_tmpl.format(1)
                full_path = os.path.join(self.root_path, record.path, file_name)

        if not self.test_mode:
            segment_indices = (
                self._sample_indices(record)
                if self.random_shift
                else self._get_val_indices(record)
            )
        else:
            segment_indices = self._get_test_indices(record)
        return self.get(record, segment_indices)

    def get(self, record, indices):

        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        process_data = self.transform(images)
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)


class customCVSDataSet(data.Dataset):
    def __init__(
        self, dirImage, pathJson, dataSetName, rule, transform, numSegments, categories
    ):
        self.dirImage = dirImage
        self.transform = transform
        self.numSegments = numSegments
        self.dataSetName = dataSetName
        self.categories = categories
        with open(pathJson, encoding="utf-8") as f:
            json_msg = json.load(f)

        labels_msg = json_msg["labels"]
        labels = sorted(list(labels_msg[rule][dataSetName].keys()))
        self.listSequences = []
        self.listLabels = []
        for label in labels:
            self.listSequences.extend(labels_msg[rule][dataSetName][label])
            self.listLabels.extend(
                [
                    label,
                ]
                * len(labels_msg[rule][dataSetName][label])
            )

    def randomShift(self, sequenceLen=24):
        interval = int(sequenceLen // self.numSegments)
        listInd = list(range(interval // 2, sequenceLen, interval))
        if self.dataSetName == "train":
            useInd = [
                random.choice(list(range(-1 * (interval // 2), interval // 2 + 1))) + x
                for x in listInd
            ]
        else:
            useInd = listInd
        return useInd

    def __len__(self):
        return len(self.listSequences)

    def __getitem__(self, index):
        sequence = self.listSequences[index]
        label = self.listLabels[index]
        choiseInd = self.randomShift()
        lastImage = sequence[-1]
        sequence = [sequence[x] for x in choiseInd]
        listImg = []
        for imgName in sequence:
            img = Image.open(os.path.join(self.dirImage, imgName)).convert("RGB")
            listImg.append(img)
        processData = self.transform(listImg)

        if self.dataSetName == "test":
            return processData, torch.tensor(self.categories.index(label)), lastImage
        else:
            return processData, torch.tensor(self.categories.index(label))


class singleCategoryDataSet(data.Dataset):
    def __init__(
        self,
        dirImage,
        jsonPath,
        dataSetName,
        transform,
        numSegments,
        categories,
        addFeature=False,
    ):
        self.dirImage = dirImage
        self.transform = transform
        self.numSegments = numSegments
        self.dataSetName = dataSetName
        self.categories = categories
        self.addFeature = addFeature
        self.MKDataSetMsg(jsonPath, dataSetName)

    def MKDataSetMsg(self, jsonPath, dataSetName):
        with open(jsonPath, encoding="utf-8") as f:
            dictMsg = json.load(f)
            f.close()

        self.listSequence = []
        self.dictSequenceIDLabels = {}

        print("格式化数据生成中...")
        for rule in dictMsg.keys():
            for score in sorted(dictMsg[rule][dataSetName].keys(), reverse=True):
                for sequence in tqdm.tqdm(
                    dictMsg[rule][dataSetName][score], desc=f"{rule}-{score}"
                ):
                    if sequence not in self.listSequence:
                        self.listSequence.append(sequence)
                    sequenceID = self.listSequence.index(sequence)
                    if sequenceID not in self.dictSequenceIDLabels:
                        self.dictSequenceIDLabels[sequenceID] = [
                            -64,
                        ]
                    ruleID = list(self.categories.keys()).index(rule)
                    scoreID = self.categories[rule].index(score)
                    self.dictSequenceIDLabels[sequenceID][ruleID] = scoreID

    def randomShift(self, sequenceLen=24):
        interval = int(sequenceLen // self.numSegments)
        listInd = list(range(interval // 2, sequenceLen, interval))
        if self.dataSetName == "train":
            useInd = [
                random.choice(list(range(-1 * (interval // 2), interval // 2 + 1))) + x
                for x in listInd
            ]
        else:
            useInd = listInd
        return useInd

    def mkInputFeature(self, listImg):
        listCanny = []
        listHOG = []
        listGray = [np.array(x.convert("L"), dtype=np.uint8) for x in listImg]

        for img in listGray:
            H, W = img.shape
            ratio = 270 / min(H, W)
            img = cv2.resize(img, (round(W * ratio), round(H * ratio)))
            gaussian = cv2.GaussianBlur(img, (5, 5), 0)
            Canny = cv2.Canny(gaussian, 50, 100)
            _, hogFeature = hog(
                img, pixels_per_cell=(20, 20), cells_per_block=(3, 3), visualize=True
            )
            listCanny.append(Image.fromarray(Canny.astype(np.uint8)).convert("L"))
            listHOG.append(Image.fromarray(hogFeature.astype(np.uint8)).convert("L"))

        postCanny = self.transform(listCanny)
        postHog = self.transform(listHOG)
        return postCanny, postHog

    def __len__(self):
        return len(self.listSequence)

    def __getitem__(self, index):
        sequence = self.listSequence[index]
        label = self.dictSequenceIDLabels[index]
        choiseInd = self.randomShift()
        lastImage = sequence[-1]
        sequence = [sequence[x] for x in choiseInd]
        listImg = []
        for imgName in sequence:
            img = Image.open(os.path.join(self.dirImage, imgName)).convert("RGB")
            listImg.append(img)

        processData = self.transform(listImg)
        if self.addFeature:
            postCanny, postHog = self.mkInputFeature(listImg)
            listConcat = []
            for i in range(len(listImg)):
                listConcat.append(processData[i * 3 : (i + 1) * 3])
                listConcat.append(postCanny[i : i + 1])
                listConcat.append(postHog[i : i + 1])
            processData = torch.cat(listConcat, 0)

        if self.dataSetName == "test":
            return processData, torch.tensor(label), lastImage
        else:
            return processData, torch.tensor(label)


class allCVSDataSet(data.Dataset):
    def __init__(
        self,
        dirImage,
        jsonPath,
        dataSetName,
        transform,
        numSegments,
        categories,
        addFeature=False,
    ):
        self.dirImage = dirImage
        self.transform = transform
        self.numSegments = numSegments
        self.dataSetName = dataSetName
        self.categories = categories
        self.addFeature = addFeature
        self.MKDataSetMsg(jsonPath, dataSetName)

    def MKDataSetMsg(self, jsonPath, dataSetName):
        with open(jsonPath, encoding="utf-8") as f:
            dictMsg = json.load(f)
            f.close()

        self.listSequence = []
        self.dictSequenceIDLabels = {}

        print("格式化数据生成中...")
        for rule in dictMsg.keys():
            for score in sorted(dictMsg[rule][dataSetName].keys(), reverse=True):
                for sequence in tqdm.tqdm(
                    dictMsg[rule][dataSetName][score], desc=f"{rule}-{score}"
                ):
                    if sequence not in self.listSequence:
                        self.listSequence.append(sequence)
                    sequenceID = self.listSequence.index(sequence)
                    if sequenceID not in self.dictSequenceIDLabels:
                        self.dictSequenceIDLabels[sequenceID] = [-64, -64, -64]
                    ruleID = list(self.categories.keys()).index(rule)
                    scoreID = self.categories[rule].index(score)
                    self.dictSequenceIDLabels[sequenceID][ruleID] = scoreID

    def randomShift(self, sequenceLen=24):
        interval = int(sequenceLen // self.numSegments)
        listInd = list(range(interval // 2, sequenceLen, interval))
        if self.dataSetName == "train":
            useInd = [
                random.choice(list(range(-1 * (interval // 2), interval // 2 + 1))) + x
                for x in listInd
            ]
        else:
            useInd = listInd
        return useInd

    def mkInputFeature(self, listImg):
        listCanny = []
        listHOG = []
        listGray = [np.array(x.convert("L"), dtype=np.uint8) for x in listImg]

        for img in listGray:
            H, W = img.shape
            ratio = 270 / min(H, W)
            img = cv2.resize(img, (round(W * ratio), round(H * ratio)))
            gaussian = cv2.GaussianBlur(img, (5, 5), 0)
            Canny = cv2.Canny(gaussian, 50, 100)
            _, hogFeature = hog(
                img, pixels_per_cell=(20, 20), cells_per_block=(3, 3), visualize=True
            )
            listCanny.append(Image.fromarray(Canny.astype(np.uint8)).convert("L"))
            listHOG.append(Image.fromarray(hogFeature.astype(np.uint8)).convert("L"))

        postCanny = self.transform(listCanny)
        postHog = self.transform(listHOG)
        return postCanny, postHog

    def __len__(self):
        return len(self.listSequence)

    def __getitem__(self, index):
        sequence = self.listSequence[index]
        label = self.dictSequenceIDLabels[index]
        choiseInd = self.randomShift()
        lastImage = sequence[-1]
        sequence = [sequence[x] for x in choiseInd]
        listImg = []
        for imgName in sequence:
            img = Image.open(os.path.join(self.dirImage, imgName)).convert("RGB")
            listImg.append(img)

        processData = self.transform(listImg)
        if self.addFeature:
            listConcat = []
            if False:
                postCanny, postHog = self.mkInputFeature(listImg)
                for i in range(len(listImg)):
                    listConcat.append(processData[i * 3 : (i + 1) * 3])
                    listConcat.append(postCanny[i : i + 1])
                    listConcat.append(postHog[i : i + 1])
            else:
                mask_folder = "/public/ss/wyx/dataset/CVSSegmentationMask/v1"
                for i, img_name in enumerate(sequence):
                    mask_path = os.path.join(
                        mask_folder, img_name.replace(".jpg", ".png")
                    )
                    mask = cv2.imread(mask_path)
                    mask = cv2.resize(
                        mask, (processData.shape[2], processData.shape[1])
                    )
                    mask = torch.from_numpy(mask[..., 0]).float()
                    mask = mask / torch.max(mask) - 0.45
                    mask = torch.unsqueeze(mask, 0)
                    listConcat.append(processData[i * 3 : (i + 1) * 3])
                    listConcat.append(mask)

            processData = torch.cat(listConcat, 0)

        if self.dataSetName == "test":
            return processData, torch.tensor(label), lastImage
        else:
            return processData, torch.tensor(label)


class allCVSBalanceDataSet:
    def __init__(
        self,
        dirImage,
        jsonPath,
        dataSetName,
        transform,
        numSegments,
        categories,
        addFeature=False,
        appearThreshold=4,
        rule=None,
    ):
        self.dirImage = dirImage
        self.transform = transform
        self.numSegments = numSegments
        self.dataSetName = dataSetName
        self.categories = categories
        self.addFeature = addFeature
        self.MKDataSetMsg(jsonPath)
        self.jumpCount = 0
        self.appearThreshold = appearThreshold
        self.rule = rule

    def MKDataSetMsg(self, jsonPath):
        print("dataSet loading...")
        with open(jsonPath, encoding="utf-8") as f:
            self.sequenceMsg = json.load(f)
            f.close()
        print("dataSet loaded!")

    def randomShift(self, sequenceLen=24):
        interval = int(sequenceLen // self.numSegments)
        listInd = list(range(interval // 2, sequenceLen, interval))
        if self.dataSetName == "train":
            useInd = [
                random.choice(list(range(-1 * (interval // 2), interval // 2 + 1))) + x
                for x in listInd
            ]
        else:
            useInd = listInd
        return useInd

    def calculateAppearCount(self, scoreSame, scoreDifferent, choiseCategory):
        sameAppearCount = sum([1 for x in scoreSame if x == choiseCategory])
        differentAppearCount = (
            sum([1 for x in scoreDifferent if x == choiseCategory]) * 0.5
        )
        appearCount = sameAppearCount + differentAppearCount

        return appearCount

    def getTrainInput(self, count, tryCount):
        listProcessData = []
        listUseLabels = []
        time0 = time.time()
        for _ in range(count):
            if self.rule is None:
                list_rule = sorted(self.sequenceMsg[self.dataSetName].keys())
            else:
                list_rule = self.rule
            for rule in list_rule:
                for score in self.sequenceMsg[self.dataSetName][rule].keys():
                    while True:
                        sequence = random.choice(
                            self.sequenceMsg[self.dataSetName][rule][score]
                        )
                        for _ in range(tryCount):
                            choiseInd = self.randomShift()
                            sequenceChoise = [sequence[x] for x in choiseInd]
                            categoryId = self.categories.index(rule)
                            scoreSame = [
                                max(x[categoryId + 1])
                                for x in sequenceChoise
                                if len(x[categoryId + 1]) == 1
                            ]
                            scoreDifferent = [
                                max(x[categoryId + 1])
                                for x in sequenceChoise
                                if len(x[categoryId + 1]) > 1
                            ]

                            scoreSame.append(-1)
                            scoreDifferent.append(-1)

                            appearCount = self.calculateAppearCount(
                                scoreSame, scoreDifferent, int(score)
                            )
                            if (
                                max(scoreSame) >= max(scoreDifferent)
                                and max(scoreSame) != -1
                                and appearCount >= self.appearThreshold
                            ):
                                break

                        if (
                            max(scoreSame) >= max(scoreDifferent)
                            and max(scoreSame) != -1
                            and appearCount >= self.appearThreshold
                        ):
                            break
                        self.jumpCount += 1
                    listLabel = []
                    for i in range(3):
                        scoreSame = [
                            max(x[i + 1]) for x in sequenceChoise if len(x[i + 1]) == 1
                        ]
                        scoreDifferent = [
                            max(x[i + 1]) for x in sequenceChoise if len(x[i + 1]) > 1
                        ]

                        scoreSame.append(-1)
                        scoreDifferent.append(-1)

                        if (
                            max(scoreSame) >= max(scoreDifferent)
                            and max(scoreSame) != -1
                        ):
                            listLabel.append(max(scoreSame))
                        else:
                            listLabel.append(-64)

                    sequenceChoise = [x[0] for x in sequenceChoise]
                    listImg = []
                    for imgName in sequenceChoise:
                        img = Image.open(os.path.join(self.dirImage, imgName)).convert(
                            "RGB"
                        )
                        listImg.append(img)
                    # self.visualSequence(listImg, listLabel)

                    processData = self.transform(listImg)
                    if self.addFeature:
                        listConcat = []
                        if False:
                            postCanny, postHog = self.mkInputFeature(listImg)
                            for i in range(len(listImg)):
                                listConcat.append(processData[i * 3 : (i + 1) * 3])
                                listConcat.append(postCanny[i : i + 1])
                                listConcat.append(postHog[i : i + 1])
                        else:
                            mask_folder = (
                                "/public/ss/wyx/dataset/CVSSegmentationMask/v1"
                            )
                            for i, img_name in enumerate(sequenceChoise):
                                mask_path = os.path.join(
                                    mask_folder, img_name.replace(".jpg", ".png")
                                )
                                mask = cv2.imread(mask_path)
                                mask = cv2.resize(
                                    mask, (processData.shape[2], processData.shape[1])
                                )
                                mask = torch.from_numpy(mask[..., 0]).float()
                                mask = mask / torch.max(mask) - 0.45
                                mask = torch.unsqueeze(mask, 0)
                                listConcat.append(processData[i * 3 : (i + 1) * 3])
                                listConcat.append(mask)

                        processData = torch.cat(listConcat, 0)

                    listProcessData.append(processData)
                    listUseLabels.append(listLabel)
        time1 = time.time()
        return torch.stack(listProcessData, 0), torch.Tensor(listUseLabels).long()

    def mkInputFeature(self, listImg):
        listCanny = []
        listHOG = []
        listGray = [np.array(x.convert("L"), dtype=np.uint8) for x in listImg]

        for img in listGray:
            H, W = img.shape
            ratio = 270 / min(H, W)
            img = cv2.resize(img, (round(W * ratio), round(H * ratio)))
            gaussian = cv2.GaussianBlur(img, (5, 5), 0)
            Canny = cv2.Canny(gaussian, 50, 100)
            _, hogFeature = hog(
                img, pixels_per_cell=(20, 20), cells_per_block=(3, 3), visualize=True
            )
            listCanny.append(Image.fromarray(Canny.astype(np.uint8)).convert("L"))
            listHOG.append(Image.fromarray(hogFeature.astype(np.uint8)).convert("L"))

        postCanny = self.transform(listCanny)
        postHog = self.transform(listHOG)
        return postCanny, postHog

    def mkTestSequence(self):
        self.testSequence = []
        self.testLabels = []
        self.cc = 0
        for rule in self.sequenceMsg[self.dataSetName]:
            for score in self.sequenceMsg[self.dataSetName][rule]:
                self.testSequence.extend(
                    self.sequenceMsg[self.dataSetName][rule][score]
                )
                label = [-64] * len(self.categories)
                label[self.categories.index(rule)] = int(score)
                self.testLabels.extend(
                    [label] * len(self.sequenceMsg[self.dataSetName][rule][score])
                )
        self.sequenceLen = len(self.testSequence)

    def getTestInput(self, count):
        listProcessData = []
        listUseLabels = []
        for i in range(self.cc, min(self.cc + count, len(self.testSequence))):
            sequence = [x[0] for x in self.testSequence[i]]
            choiseInd = self.randomShift()
            sequence = [sequence[x] for x in choiseInd]
            label = self.testLabels[i]
            listImg = []
            for imgName in sequence:
                img = Image.open(os.path.join(self.dirImage, imgName)).convert("RGB")
                listImg.append(img)
            processData = self.transform(listImg)
            listProcessData.append(processData)
            listUseLabels.append(label)

        self.cc = min(self.cc + count, len(self.testSequence))
        if self.cc == len(self.testSequence):
            return (
                torch.stack(listProcessData, 0),
                torch.Tensor(listUseLabels).long(),
                False,
            )
        else:
            return (
                torch.stack(listProcessData, 0),
                torch.Tensor(listUseLabels).long(),
                True,
            )

    def visualSequence(self, listImg, labels):
        listImg = [np.array(x, dtype=np.uint8) for x in listImg]
        imgSave = np.concatenate(
            (
                np.concatenate((listImg[:4]), axis=1),
                np.concatenate((listImg[4:]), axis=1),
            ),
            axis=0,
        )
        imgSave = cv2.cvtColor(imgSave, cv2.COLOR_RGB2BGR)
        saveFolder = r"D:\work\temporary\20210530\{}_{}_{}".format(
            labels[0], labels[1], labels[2]
        )
        os.makedirs(saveFolder, exist_ok=True)
        savePath = os.path.join(
            saveFolder, "{:0>3}.jpg".format(len(os.listdir(saveFolder)))
        )
        cv2.imwrite(savePath, imgSave)


if __name__ == "__main__":
    import torchvision
    from ops.transforms import *

    imageFolder = r"D:\work\dataSet\CVS\for sequence\CVSSequenceFrames20210610"
    jsonPath = r"D:\work\dataSet\CVS\score\v1\sequenceAllNoPublic_24_8_4.json"
    """
    transforms.RandomRotation                                                --随机旋转
    transforms.RandomAffine                                                  --仿射变换
    transforms.RandomApply(transforms, p=0.5)                                --给transform加执行概率
    transforms.RandomPerspective(distortion_scale=0.6, p=1.0)                --随机透视
    transforms.RandomEqualize()                                              --随机色彩直方图变换

    """

    normalize = IdentityTransform()
    tf = torchvision.transforms.Compose(
        [
            GroupAugmentation(),
            GroupPad(),
            GroupScale(int(224)),
            Stack(roll=(False)),
            ToTorchFormatTensor(div=(True)),
            normalize,
        ]
    )

    dataset = customCVSDataSet(
        "",
        r"D:\work\dataSet\LCPhase\recognition_lc_phase\v1\1\v1_1.json",
        "test",
        "phase",
        tf,
        8,
        [0, 1, 2, 3, 4, 5, 6],
    )
    loader = torch.utils.data.DataLoader(dataset, 2, False)
    for processData, label in loader:
        stop = 0
