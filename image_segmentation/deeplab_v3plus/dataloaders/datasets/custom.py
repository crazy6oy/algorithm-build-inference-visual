import numpy as np
import torch
from torch.utils.data import Dataset
from mypath import Path
from tqdm import trange
import os
from torchvision import transforms
from dataloaders import custom_transforms as tr
from PIL import Image, ImageFile


class customSegmentation(Dataset):
    CAT_LIST = [0, 1, 2, 3, 4, 5, 6]
    NUM_CLASSES = len(CAT_LIST)

    def __init__(self, args, split):
        self.args = args
        self.dataSetRecordPath = args.dataSetRecordPath
        self.split = split
        self.imageFolder = os.path.join(args.imageFolder, self.split, "image")
        self.maskFolder = os.path.join(args.imageFolder, self.split, "mask")

        if self.dataSetRecordPath == None:
            listMaskName = os.listdir(self.maskFolder)
        elif self.dataSetRecordPath.split(".")[-1] == "json":
            pass
        else:
            with open(self.dataSetRecordPath, encoding="utf-8") as f:
                listMaskName = f.readlines()
                f.close()
            listMaskName = [x.strip() for x in listMaskName]
        self.listMaskName = listMaskName

    def __len__(self):
        return len(self.listMaskName)

    def __getitem__(self, index):
        maskName = self.listMaskName[index]
        imgName = maskName.replace(".png", ".jpg")

        imgPath = os.path.join(self.imageFolder, imgName)
        maskPath = os.path.join(self.maskFolder, maskName)

        img = Image.open(imgPath).convert("RGB")
        mask = Image.open(maskPath).convert("L")
        sample = {"image": img, "label": mask}

        if self.split == "train":
            return self.transform_tr(sample), imgName
        else:
            return self.transform_val(sample), imgName

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose(
            [
                tr.RandomHorizontalFlip(),
                tr.RandomScaleCrop(
                    base_size=self.args.base_size, crop_size=self.args.crop_size
                ),
                tr.RandomGaussianBlur(),
                tr.Normalize(),
                tr.ToTensor(),
            ]
        )

        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose(
            [tr.Pad(crop_size=self.args.crop_size), tr.Normalize(), tr.ToTensor()]
        )

        return composed_transforms(sample)


if __name__ == "__main__":
    from PIL import Image

    img = Image.open(r"C:\Users\wangyx\Desktop\test_img.jpg")

    def transform_val(sample):
        composed_transforms = transforms.Compose(
            [
                tr.FixScaleCrop(crop_size=512),
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor(),
            ]
        )

        return composed_transforms(sample)

    result = transform_val({"image": img, "label": img})
    min_x = torch.min(result["image"])
    max_x = torch.max(result["image"])
    stop = 0
