import os
import time
import shutil
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_

import numpy as np
import tqdm
import json
from ops.dataset import TSNDataSet, customCVSDataSet, allCVSDataSet
from ops.models import TSN
from ops.transforms import *
from opts import parser
from ops import dataset_config
from ops.utils import AverageMeter
from ops.temporal_shift import make_temporal_pool

import cv2

from tensorboardX import SummaryWriter

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    dictCategories = {
        "cvsi": ["-1", "0", "1", "2"],
        "cvsii": ["-1", "0", "2"],
        "cvsiii": ["-1", "0", "1", "2"],
    }
    imageFolder = "/home/withai/wangyx/temp/CVSProject/outputs"
    sequenceJsonPath = "/home/withai/wangyx/darknet/data/cvs_region/test.json"
    recordFolder = "/home/withai/wangyx/darknet/data/cvs_region/czx/cvsiii"
    os.makedirs(recordFolder, exist_ok=True)
    num_class = 0
    for rule in dictCategories:
        num_class += len(dictCategories[rule])
    args.resume = "/home/withai/wangyx/darknet/data/cvs_region/CVSOnlyCVSIIINoPublicRegion24_8_4MAPBest.pth.tar"

    full_arch_name = args.arch
    if args.shift:
        full_arch_name += "_shift{}_{}".format(args.shift_div, args.shift_place)
    if args.temporal_pool:
        full_arch_name += "_tpool"
    args.store_name = "_".join(
        [
            "TSM",
            args.dataset,
            args.modality,
            full_arch_name,
            args.consensus_type,
            "segment%d" % args.num_segments,
            "e{}".format(args.epochs),
        ]
    )
    if args.pretrain != "imagenet":
        args.store_name += "_{}".format(args.pretrain)
    if args.lr_type != "step":
        args.store_name += "_{}".format(args.lr_type)
    if args.dense_sample:
        args.store_name += "_dense"
    if args.non_local > 0:
        args.store_name += "_nl"
    if args.suffix is not None:
        args.store_name += "_{}".format(args.suffix)
    print("storing name: " + args.store_name)

    model = TSN(
        num_class,
        args.num_segments,
        args.modality,
        base_model=args.arch,
        consensus_type=args.consensus_type,
        dropout=args.dropout,
        img_feature_dim=args.img_feature_dim,
        partial_bn=not args.no_partialbn,
        pretrain=args.pretrain,
        is_shift=args.shift,
        shift_div=args.shift_div,
        shift_place=args.shift_place,
        fc_lr5=not (args.tune_from and args.dataset in args.tune_from),
        temporal_pool=args.temporal_pool,
        non_local=args.non_local,
    )

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    if os.path.isfile(args.resume):
        print(("=> loading checkpoint '{}'".format(args.resume)))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint["epoch"]
        best_prec1 = checkpoint["mAP"]
        model.load_state_dict(checkpoint["state_dict"])
        print(
            (
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.evaluate, checkpoint["epoch"]
                )
            )
        )
    else:
        print(("=> no checkpoint found at '{}'".format(args.resume)))
        exit()

    if args.tune_from:
        print(("=> fine-tuning from '{}'".format(args.tune_from)))
        sd = torch.load(args.tune_from)
        sd = sd["state_dict"]
        model_dict = model.state_dict()
        replace_dict = []
        for k, v in sd.items():
            if k not in model_dict and k.replace(".net", "") in model_dict:
                print("=> Load after remove .net: ", k)
                replace_dict.append((k, k.replace(".net", "")))
        for k, v in model_dict.items():
            if k not in sd and k.replace(".net", "") in sd:
                print("=> Load after adding .net: ", k)
                replace_dict.append((k.replace(".net", ""), k))

        for k, k_new in replace_dict:
            sd[k_new] = sd.pop(k)
        keys1 = set(list(sd.keys()))
        keys2 = set(list(model_dict.keys()))
        set_diff = (keys1 - keys2) | (keys2 - keys1)
        print("#### Notice: keys that failed to load: {}".format(set_diff))
        if args.dataset not in args.tune_from:  # new dataset
            print("=> New dataset, do not load fc weights")
            sd = {k: v for k, v in sd.items() if "fc" not in k}
        if args.modality == "Flow" and "Flow" not in args.tune_from:
            sd = {k: v for k, v in sd.items() if "conv1.weight" not in k}
        model_dict.update(sd)
        model.load_state_dict(model_dict)

    if args.temporal_pool and not args.resume:
        make_temporal_pool(model.module.base_model, args.num_segments)

    cudnn.benchmark = True

    # Data loading code
    if args.modality != "RGBDiff":
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    valTransformer = torchvision.transforms.Compose(
        [
            GroupPad(),
            GroupScale(int(scale_size)),
            # GroupCenterCrop(crop_size),
            Stack(roll=(args.arch in ["BNInception", "InceptionV3"])),
            ToTorchFormatTensor(div=(args.arch not in ["BNInception", "InceptionV3"])),
            normalize,
        ]
    )

    val_loader = torch.utils.data.DataLoader(
        allCVSDataSet(
            imageFolder,
            sequenceJsonPath,
            "test",
            valTransformer,
            args.num_segments,
            dictCategories,
        ),
        batch_size=4,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    # define loss function (criterion) and optimizer
    if args.loss_type == "nll":
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")

    validate(val_loader, model, criterion, dictCategories, recordFolder)


def validate(val_loader, model, criterion, dictCategories, recordFolder):
    batch_time = AverageMeter()

    # switch to evaluate mode
    model.eval()

    # get layer
    resNet = model.module.base_model
    resNet0 = resNet.conv1
    resNet1 = resNet.bn1
    resNet2 = resNet.relu
    resNet3 = resNet.maxpool
    resNet4 = resNet.layer1
    resNet5 = resNet.layer2
    resNet6 = resNet.layer3
    resNet7 = resNet.layer4
    resNet8 = resNet.avgpool
    resNet9 = resNet.fc
    fc = model.module.new_fc
    consensus = model.module.consensus

    fcParam = fc.state_dict()["weight"].detach().cpu().tolist()
    # maxIndex = torch.argsort(fcParam, 1, True)

    with torch.no_grad():
        for i, (input, target, imagesName) in enumerate(tqdm.tqdm(val_loader)):
            target = target.cuda()

            inputTest = input.view((-1, 3) + input.size()[-2:]).cuda()
            out = resNet0(inputTest)
            out = resNet1(out)
            out = resNet2(out)
            out = resNet3(out)
            out = resNet4(out)
            out = resNet5(out)
            out = resNet6(out)
            featureMap = resNet7(out)
            out = resNet8(featureMap)
            out = resNet9(out)
            out = out[:, :, 0, 0]
            out = fc(out)
            out = out.view((-1, 8) + out.size()[1:])
            out = consensus(out)
            out = out[:, 0]

            # featureMapArray = featureMap.detach().cpu().numpy()
            # featureMap01 = np.expand_dims([0, 0], 2).repeat(3, 2)
            # hp = normHeatmap2RGBHeatmap(featureMap01)

            # save feature map
            featureMap = (
                torch.reshape(featureMap, (len(imagesName), 8, 2048, 8, 8))
                .detach()
                .cpu()
                .numpy()
            )
            for num, imageName in enumerate(imagesName):
                feature_map_single_frame = featureMap[num, -1]
                weight = np.expand_dims(
                    np.expand_dims(np.array(fcParam[torch.argmax(out[num][7:])]), 1), 2
                )

                fp = np.average(feature_map_single_frame * weight, 0)
                fp = np.expand_dims(fp, 2).repeat(3, 2)
                fp = normHeatmap2RGBHeatmap(fp)

                # if "HX" in imageName:
                #     fp = cv2.resize(fp, (1920, 1920))
                # else:
                #     fp = cv2.resize(fp, (720, 720))
                fp = cv2.cvtColor(fp, cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    os.path.join(recordFolder, imagesName[num].replace(".jpg", ".png")),
                    fp,
                )
    return 0


def normHeatmap2RGBHeatmap(normHeatmap):
    """
    params
        normHeatmap:进行了全图归一化的三通道热图，通道顺序R G B
    """

    minValues = np.min(normHeatmap)
    maxValues = np.max(normHeatmap)
    normHeatmap = (normHeatmap - minValues) / (maxValues - minValues)

    normHeatmap[..., 0] = normHeatmap[..., 0] * 510 - 255
    normHeatmap[..., 1] = (
        -4 * normHeatmap[..., 1] * normHeatmap[..., 1] + 4 * normHeatmap[..., 1]
    ) * 255
    normHeatmap[..., 2] = normHeatmap[..., 2] * (-510) + 255
    normHeatmap[normHeatmap < 0] = 0
    normHeatmap[normHeatmap > 255] = 255
    return normHeatmap.astype(np.uint8)


if __name__ == "__main__":
    main()
