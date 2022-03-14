import argparse
import os
import time
import shutil
import torch
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
from torchvision import transforms
import numpy as np
import tqdm
from PIL import Image
import json


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
        return torch.argmax(output).item()

    def Main(
        self,
        videopath,
        fpsUse=1,
    ):
        reader = imageio.get_reader(videopath)
        videoMsg = reader.get_meta_data()
        videoFps = videoMsg["fps"]
        videoDuration = videoMsg["duration"]
        videoTime = round(videoFps * videoDuration)

        postResult = {}
        # frameGroup = []
        for i, frame in enumerate(
            tqdm.tqdm(
                reader, desc=videopath.split("/")[-1].split("_")[0], total=videoTime
            )
        ):
            secondLastFrame = int((i - 1) // (videoFps / fpsUse))
            secondNowFrame = int(i // (videoFps / fpsUse))
            if secondLastFrame < secondNowFrame:
                # if len(frameGroup) < 24:
                #     pass
                # elif len(frameGroup) == 24:
                #     frameGroup = frameGroup[1:]
                # else:
                #     raise ValueError('!!!')
                frameGroup = [
                    frame,
                ]
                if len(frameGroup) == 1:
                    inputFrames = self.dealFrame(frameGroup)
                    output = self.modelCalculate(inputFrames)
                    postResult[secondNowFrame] = self.dealModelOutput(output)

        operationName = videopath.split("/")[-1].split("_")[0]
        with open(f"{self.saveDir}/{operationName}.json", "w", encoding="utf-8") as f:
            json.dump(postResult, f, ensure_ascii=False, indent=2)
            f.close()

    def dealFrame(self, frameGroup):
        frameInput = []
        if self.modelName == "TRN":
            frameGroup = [
                Image.fromarray(x).convert("RGB")
                for i, x in enumerate(frameGroup)
                if i in [1, 4, 7, 10, 13, 16, 19, 22]
            ]
            inputFrames = transformer(frameGroup)
            inputFrames = torch.unsqueeze(inputFrames, 0)
        elif self.modelName == "EfficientNet":
            for i, img in enumerate(frameGroup):
                H, W, C = img.shape
                img = Image.fromarray(img.astype("uint8")).convert("RGB")
                out0 = self.transformer(img)
                pad_s = (
                    ((H - W) // 2, H - W - (H - W) // 2, 0, 0)
                    if H > W
                    else (0, 0, (W - H) // 2, W - H - (W - H) // 2)
                )
                out1 = torch.nn.functional.pad(out0, pad_s, value=0)
                out2 = torch.nn.functional.interpolate(
                    out1.unsqueeze(0), size=380, mode="area"
                )[0]
                frameInput.append(out2)
            inputFrames = torch.stack(frameInput, 0)
        else:
            raise NameError("modelName Error!")
        return inputFrames.cuda()


def go(videopath, checkPointPath, saveDir):
    modelName = "EfficientNet"
    videopath = videopath
    checkPointPath = checkPointPath
    transformer = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    tool = Calculate(modelName, transformer, checkPointPath, saveDir)
    tool.Main(videopath)


if __name__ == "__main__":
    modelName = "EfficientNet"
    videopath = "/mnt/video/LC10000/CompleteVideo/hospital_id=1/surgery_id=783/video/20210203-LC-HX-0033834527_HD1080.mp4"
    checkPointPath = (
        "/home/withai/wangyx/checkpoint/efficientNet/efficientnet-b4_CVSI.pth"
    )
    transformer = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    tool = Calculate(modelName, transformer, checkPointPath)
    tool.Main(videopath)
