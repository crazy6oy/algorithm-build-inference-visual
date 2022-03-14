import os
import time
import shutil
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_

import tqdm
from ops.dataset import TSNDataSet, customCVSDataSet
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
        "cvsi": ["-1", "0", "1", "2"],
        "cvsii": ["-1", "0", "2"],
        "cvsiii": ["-1", "0", "1", "2"],
    }
    dirImage = "D:/work/dataSet/CVS/for sequence/AllImage"
    pathJson = "/home/withai/Pictures/sequence.json"
    rule = "cvsiii"
    categories = dictCategories[rule]
    num_class = len(categories)
    args.resume = f"/home/withai/wangyx/checkPoint/TSM/{rule}.pth.tar"

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
        best_prec1 = checkpoint["best_prec1"]
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

    val_loader = torch.utils.data.DataLoader(
        customCVSDataSet(
            dirImage,
            pathJson,
            "test",
            rule,
            torchvision.transforms.Compose(
                [
                    GroupScale(int(scale_size)),
                    GroupCenterCrop(crop_size),
                    Stack(roll=(args.arch in ["BNInception", "InceptionV3"])),
                    ToTorchFormatTensor(
                        div=(args.arch not in ["BNInception", "InceptionV3"])
                    ),
                    normalize,
                ]
            ),
            args.num_segments,
            categories,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    # define loss function (criterion) and optimizer
    if args.loss_type == "nll":
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")

    prec1 = validate(val_loader, model, criterion)


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    calLabel = []
    calResult = []
    calConf = []
    dictOutput = {}
    with torch.no_grad():
        for i, (input, target, imagesName) in enumerate(tqdm.tqdm(val_loader)):
            target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            (prec1, prec5), targetTemp, resultTemp, confTemp = accuracy(
                output.data, target, topk=(1, 3)
            )
            calLabel.append(targetTemp)
            calResult.append(resultTemp)
            calConf.append(confTemp)

            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            for num, imageName in enumerate(imagesName):
                imageName = imageName.split(".")[0]
                dictOutput[imageName] = [
                    torch.argmax(output[num], 0).item(),
                    target[num].item(),
                ]

    output = "Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}".format(
        top1=top1, top5=top5, loss=losses
    )
    print(output)
    print(dictOutput)

    labels = torch.cat(calLabel, 0).cpu().detach()
    results = torch.cat(calResult, 0).cpu().detach()
    conf = torch.cat(calConf, 0).cpu().detach()
    tp = (labels == results).long()
    getConfusionMatrix(tp.numpy(), labels.numpy(), results.numpy(), conf.numpy())

    return top1.avg


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


if __name__ == "__main__":
    gpuid = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpuid
    main()
