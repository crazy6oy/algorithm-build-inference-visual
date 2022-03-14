import argparse
import os
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm
import torch.nn.functional as F
import multiprocessing as mp
import imageio

# from utils.TRN import GroupScale, GroupCenterCrop, Stack, ToTorchFormatTensor, normalizeFunc
from ops.transforms import *
from cfg.TSM import scaleSize, cropSize, arch, modality
import numpy as np
import tqdm
from PIL import Image
import json
import cv2


class Calculate:
    def __init__(self, modelName, transformer, checkPointPath, saveDir):
        self.modelName = modelName
        self.transformer = transformer
        self.modelInit(checkPointPath)
        self.saveDir = saveDir

    def modelInit(self, checkPointPath):
        if not os.path.isfile(checkPointPath):
            raise FileNotFoundError("model path Error!")
        print(checkPointPath)
        self.model = torch.load(checkPointPath)
        self.model.cuda()
        self.model.eval()
        # self.input_mean = self.model.input_mean
        # self.input_std = self.model.input_std
        print("loaded!")

    def modelCalculate(self, input):
        output = self.model(input)
        return output

    def dealModelOutput(self, output):
        return [
            torch.argmax(output[:, :4]).item(),
            torch.argmax(output[:, 4:7]).item(),
            torch.argmax(output[:, 7:]).item(),
        ]

    def Main(self, videopath, fpsUse=1):
        reader = imageio.get_reader(videopath)
        videoMsg = reader.get_meta_data()
        videoFps = videoMsg["fps"]
        videoDuration = videoMsg["duration"]
        videoTime = round(videoFps * videoDuration)

        postResult = {}
        frameGroup = []
        for i, frame in enumerate(
            tqdm.tqdm(
                reader, desc=videopath.split("/")[-1].split("_")[0], total=videoTime
            )
        ):
            imgShow = frame

            secondLastFrame = int((i - 1) / (videoFps / fpsUse))
            secondNowFrame = int(i / (videoFps / fpsUse))
            if secondLastFrame < secondNowFrame:
                if len(frameGroup) < 24:
                    pass
                elif len(frameGroup) == 24:
                    frameGroup = frameGroup[1:]
                else:
                    raise ValueError("!!!")
                frameGroup.append(frame)
                if len(frameGroup) == 24:
                    inputFrames = self.dealFrame(frameGroup, self.transformer)
                    output = self.modelCalculate(inputFrames)
                    # postResult[secondNowFrame] = self.dealModelOutput(output)
                    postResult[secondNowFrame] = torch.argmax(output).item()
                    print(f"{int(i//25)} {torch.argmax(output).item()}")

            if postResult == {}:
                labelT = -2
            else:
                labelT = postResult[secondNowFrame]
                # print(postResult[secondNowFrame])
            cv2.putText(
                imgShow,
                f"{labelT}",
                (2, 64),
                cv2.FONT_HERSHEY_COMPLEX,
                2,
                (218, 218, 218),
                2,
            )
            imgShow = cv2.cvtColor(imgShow, cv2.COLOR_RGB2BGR)
            imgShow = cv2.resize(imgShow, (960, 540))
            cv2.imshow("player", imgShow)
            cv2.waitKey(int(1000 / videoFps))

        operationName = videopath.split("/")[-1].split("_")[0]
        with open(f"{self.saveDir}/{operationName}.json", "w", encoding="utf-8") as f:
            json.dump(postResult, f, ensure_ascii=False, indent=2)
            f.close()

    def dealFrame(self, frameGroup, transformer):
        if self.modelName == "TRN":
            frameGroup = [
                Image.fromarray(x).convert("RGB")
                for i, x in enumerate(frameGroup)
                if i in [1, 4, 7, 10, 13, 16, 19, 22]
            ]
            inputFrames = transformer(frameGroup)
            inputFrames = torch.unsqueeze(inputFrames, 0)
        elif self.modelName == "TSM":
            frameGroup = [
                Image.fromarray(x).convert("RGB")
                for i, x in enumerate(frameGroup)
                if i in [1, 4, 7, 10, 13, 16, 19, 22]
            ]
            inputFrames = transformer(frameGroup)
            inputFrames = torch.unsqueeze(inputFrames, 0)
        else:
            raise NameError("modelName Error!")
        return inputFrames.cuda()


def go(videopath, checkPointPath, saveDir):
    modelName = "TSM"
    videopath = videopath
    checkPointPath = checkPointPath
    saveDir = saveDir
    normalize = GroupNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transformer = torchvision.transforms.Compose(
        [
            GroupPad(),
            GroupScale(int(scaleSize)),
            GroupCenterCrop(cropSize),
            Stack(roll=(arch in ["BNInception", "InceptionV3"])),
            ToTorchFormatTensor(div=(arch not in ["BNInception", "InceptionV3"])),
            normalize,
        ]
    )

    tool = Calculate(modelName, transformer, checkPointPath, saveDir)
    tool.Main(videopath)


if __name__ == "__main__":
    videopath = r"Z:\LC10000\CompleteVideo\hospital_id=14\surgery_id=639\video\20210111-LC-YR-WCH-TBL-N2_HD1080.mp4"
    checkPointPath = (
        r"D:\work\temporary\20210618\CVSOnlyCVSIIINoPublicRegion24_8_4MAPBest.pth"
    )
    saveDir = r"D:\work\temporary\20210604"
    go(videopath, checkPointPath, saveDir)
