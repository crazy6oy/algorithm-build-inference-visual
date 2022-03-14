import os
import time
import shutil
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_

import json
import tqdm
from ops.dataset import TSNDataSet, customCVSDataSet, allCVSDataSet
from ops.models import TSN
from ops.transforms import *
from opts import parser
from ops import dataset_config
from ops.utils import AverageMeter
from ops.temporal_shift import make_temporal_pool

from tensorboardX import SummaryWriter

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    dictCategories = {
        "cvsi": ["0", "1", "2", "3"],
        "cvsii": ["0", "1", "2"],
        "cvsiii": ["0", "1", "2", "3"],
    }
    imageFolder = (
        r"D:\work\dataSet\CVS\for sequence\CVSSequenceFramesOnlyCVSRegion20210610"
    )
    sequenceJsonPath = (
        r"D:\work\dataSet\CVS\score\v1\9\sequenceAllNoPublic_24_8_4(oldFormat).json"
    )

    dropOutId = "CVSOnlyCVSIIIRegion"
    recordJson = f"./outputs/CVSOutputDropOut_{dropOutId}.json"
    num_class = 0
    for rule in dictCategories:
        num_class += len(dictCategories[rule])
    args.resume = r"D:\work\dataSet\CVS\score\v1\9\CVSOnlyCVSIIINoPublicRegion24_8_4MAPBest.pth.tar"

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

    transformer = torchvision.transforms.Compose(
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
            transformer,
            args.num_segments,
            dictCategories,
        ),
        batch_size=2,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    # define loss function (criterion) and optimizer
    if args.loss_type == "nll":
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")

    prec1 = validate(val_loader, model, dictCategories, recordJson, dropOutId)


def validate(val_loader, model, dictCategories, recordJson, dropOutId):
    batch_time = AverageMeter()

    # switch to evaluate mode
    model.eval()

    calLabel = [[], [], []]
    calResult = [[], [], []]
    calConf = [[], [], []]
    dictOutput = {}
    with torch.no_grad():
        for i, (input, target, imagesName) in enumerate(tqdm.tqdm(val_loader)):
            # if dropOutId is None:
            #     pass
            # else:
            #     input[:, 0 * 3:6 * 3] = 0
            #     input[:, 7 * 3:8 * 3] = 0
            target = target.cuda()

            # compute output
            output = model(input)

            # measure accuracy and record loss
            output0 = output[:, :4]
            target0 = target[:, 0]
            output1 = output[:, 4:7]
            target1 = target[:, 1]
            output2 = output[:, 7:]
            target2 = target[:, 2]
            listOutputTarget = [
                [output0, target0],
                [output1, target1],
                [output2, target2],
            ]
            for num, (outputT, targetT) in enumerate(listOutputTarget):
                outputT = outputT[targetT != -64]
                imagesNameT = [y for x, y in enumerate(imagesName) if targetT[x] != -64]
                targetT = targetT[targetT != -64]
                if targetT.shape[0] == 0:
                    continue
                (_, _), targetTemp, resultTemp, confTemp = accuracy(
                    outputT.data, targetT, topk=(1, 3)
                )
                calLabel[num].append(targetTemp.cpu())
                calResult[num].append(resultTemp.cpu())
                calConf[num].append(confTemp.cpu())

                for j, imageName in enumerate(imagesNameT):
                    imageName = imageName.split(".")[0]
                    if imageName not in dictOutput.keys():
                        dictOutput[imageName] = {}
                    dictOutput[imageName][list(dictCategories.keys())[num]] = [
                        resultTemp[j].item(),
                        targetTemp[j].item(),
                        confTemp[j].item(),
                    ]

    with open(recordJson, "w", encoding="utf-8") as f:
        json.dump(dictOutput, f, ensure_ascii=False, indent=2)
        f.close()
    mAP = []
    for i in range(len(calResult)):
        print(list(dictCategories.keys())[i])
        labels = torch.cat(calLabel[i], 0).cpu().detach()
        results = torch.cat(calResult[i], 0).cpu().detach()
        conf = torch.cat(calConf[i], 0).cpu().detach()
        tp = (labels == results).long()
        aps = getConfusionMatrix(
            tp.numpy(), labels.numpy(), results.numpy(), conf.numpy()
        )
        mAP.extend(aps)

    print(f"mAP: {sum(mAP) / len(mAP)}")

    return sum(mAP) / len(mAP)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    output = torch.nn.Softmax(1)(output)

    conf, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res, target, pred[0], conf[:, 0]


from calculateAP import ap_per_class


def getConfusionMatrix(tp, labels, results, conf):
    p, r, ap, f1, classes = ap_per_class(tp, conf, results, labels)
    print(f"{classes.tolist()}\n{p.tolist()}\n{r.tolist()}\n{ap.tolist()}\n")
    return ap.tolist()


if __name__ == "__main__":
    # gpuid = "0"
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpuid
    main()
