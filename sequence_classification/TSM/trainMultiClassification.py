# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import os
import time
import shutil
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_
import numpy as np

from ops.dataset import (
    TSNDataSet,
    customCVSDataSet,
    allCVSDataSet,
    allCVSBalanceDataSet,
)
from ops.models import TSN
from ops.transforms import *
from opts import parser
from ops import dataset_config
from ops.utils import AverageMeter
from ops.temporal_shift import make_temporal_pool

from tensorboardX import SummaryWriter
from calculateAP import ap_per_class

best_mAP = 0
best_variance = 0.25
best_variance_mAP = 0

"""
Attention!!!
如果要训练所有类别中几个类别则需要根据具体情况修改下列内容：
    calculateLossACC
    useAps
    allCVSBalanceDataSet中rule
"""


def main():
    global args, best_mAP, best_variance, best_variance_mAP
    args = parser.parse_args()

    # num_class, args.train_list, args.val_list, args.root_path, prefix = dataset_config.return_dataset(args.dataset,
    #                                                                                                   args.modality)
    # args.root_path = "something-something"
    dictCategories = {
        "cvsi": ["0", "1", "2", "3"],
        "cvsii": ["0", "1", "2"],
        "cvsiii": ["0", "1", "2", "3"],
    }
    categories = ["cvsi", "cvsii", "cvsiii"]
    imageFolder = r"D:\work\Data\CVSScore\fix1"
    trainSequenceJsonPath = r"D:\work\dataSet\CVS\score\v1\sequence_24_8_1.json"
    valSequenceJsonPath = (
        r"D:\work\dataSet\CVS\score\v1\sequenceTest_24_8_1(oldFormat).json"
    )

    logFileName = "CVSTest"

    num_class = 0
    for rule in dictCategories:
        num_class += len(dictCategories[rule])

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

    logPath = os.path.join(args.root_log, args.store_name, f"{logFileName}Log.csv")
    configRecordPath = os.path.join(
        args.root_log, args.store_name, f"{logFileName}Args.txt"
    )
    mapMaxFilePath = "%s/%s/%sMAPBest.pth.tar" % (
        args.root_model,
        args.store_name,
        logFileName,
    )
    balenceMaxFilePath = "%s/%s/%sBalenceBest.pth.tar" % (
        args.root_model,
        args.store_name,
        logFileName,
    )
    lastFilePath = "%s/%s/%sLast.pth.tar" % (
        args.root_model,
        args.store_name,
        logFileName,
    )
    mapMaxModelPath = "%s/%s/%sMAPBest.pth" % (
        args.root_model,
        args.store_name,
        logFileName,
    )
    balenceMaxModelPath = "%s/%s/%sBalenceBest.pth" % (
        args.root_model,
        args.store_name,
        logFileName,
    )
    lastModelPath = "%s/%s/%sLast.pth" % (args.root_model, args.store_name, logFileName)

    check_rootfolders()

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
    policies = model.get_optim_policies()
    train_augmentation = model.get_augmentation(
        flip=False if "something" in args.dataset or "jester" in args.dataset else True
    )
    # model.base_model.conv1 = torch.nn.Conv2d(4, 64, (3, 3), (2, 2), (3, 3), bias=False)

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    optimizer = torch.optim.SGD(
        policies, args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    if args.resume:
        if args.temporal_pool:  # early temporal pool so that we can load the state_dict
            make_temporal_pool(model.module.base_model, args.num_segments)
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint["epoch"]
            best_mAP = checkpoint["mAP"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                (
                    "=> loaded checkpoint '{}' (epoch {})".format(
                        args.evaluate, checkpoint["epoch"]
                    )
                )
            )
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

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

    if args.modality == "RGB":
        data_length = 1
    elif args.modality in ["Flow", "RGBDiff"]:
        data_length = 5

    trainTransformer = torchvision.transforms.Compose(
        [
            GroupAugmentation(),
            GroupPad(),
            GroupScale(int(scale_size)),
            # GroupCenterCrop(crop_size),
            Stack(roll=(args.arch in ["BNInception", "InceptionV3"])),
            ToTorchFormatTensor(div=(args.arch not in ["BNInception", "InceptionV3"])),
            normalize,
        ]
    )
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

    train_loader = allCVSBalanceDataSet(
        imageFolder,
        trainSequenceJsonPath,
        "train",
        trainTransformer,
        args.num_segments,
        categories,
        rule="cvsiii",
    )

    # val_loader = allCVSBalanceDataSet(imageFolder,
    #                                   sequenceJsonPath,
    #                                   "valid",
    #                                   torchvision.transforms.Compose([
    #                                       GroupPad(),
    #                                       GroupScale(int(scale_size)),
    #                                       # GroupCenterCrop(crop_size),
    #                                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
    #                                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
    #                                       normalize,
    #                                   ]),
    #                                   args.num_segments, categories)

    val_loader = torch.utils.data.DataLoader(
        allCVSDataSet(
            imageFolder,
            valSequenceJsonPath,
            "valid",
            valTransformer,
            args.num_segments,
            dictCategories,
        ),
        batch_size=16,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    # define loss function (criterion) and optimizer
    if args.loss_type == "nll":
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")

    for group in policies:
        print(
            (
                "group: {} has {} params, lr_mult: {}, decay_mult: {}".format(
                    group["name"],
                    len(group["params"]),
                    group["lr_mult"],
                    group["decay_mult"],
                )
            )
        )

    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        return

    log_training = open(logPath, "w")
    with open(configRecordPath, "w") as f:
        f.write(str(args))
    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))
    cc = 0
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_type, args.lr_steps)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, log_training, tf_writer)

        # evaluate on validation set
        aps = validate(val_loader, model, criterion, epoch, log_training, tf_writer)
        mAP = sum(aps) / len(aps)
        variance = np.var(aps).item()

        # remember best prec@1 and save checkpoint
        modelState = {
            "epoch": epoch + 1,
            "arch": args.arch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "variance": variance,
            "mAP": best_mAP,
        }
        if mAP > best_mAP:
            save_checkpoint(modelState, model, mapMaxFilePath, mapMaxModelPath)
            best_mAP = max(mAP, best_mAP)
            cc = -1
        if mAP >= (best_variance_mAP - 0.03) and round(variance, 4) <= round(
            best_variance, 4
        ):
            save_checkpoint(modelState, model, balenceMaxFilePath, balenceMaxModelPath)
            best_variance = min(variance, best_variance)
            best_variance_mAP = max(mAP, best_variance_mAP)
            cc = -1
        save_checkpoint(modelState, model, lastFilePath, lastModelPath)
        cc += 1
        if cc == 8:
            checkpoint = torch.load(mapMaxFilePath)
            model.load_state_dict(checkpoint["state_dict"])
        if cc == 16:
            checkpoint = torch.load(balenceMaxFilePath)
            model.load_state_dict(checkpoint["state_dict"])
        if cc == 33:
            break

        tf_writer.add_scalar("mAP/test_top1_best", best_mAP, epoch)
        best_variance = max(variance, best_variance)
        tf_writer.add_scalar("variance/test_top1_best", best_mAP, epoch)

        output_best = "Best mAP@1: %.3f\n" % (best_mAP)
        print(output_best)
        log_training.write(output_best + "\n")
        log_training.flush()

    print("train over!")


def calculateCorrect(output, target):
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correctNum = correct[:1].reshape(-1).float().sum(0)
    return correctNum


def calculateLossACC(output, target, criterion):
    loss = 0
    correctNum = 0
    targetCount = 0
    # 需要学习CVS某条规则就开放对应部分的loss计算
    if False:
        output01 = output[:, :4][target[:, 0] != -64]
        target01 = target[:, :1][target[:, 0] != -64].reshape(-1)
        if output01.shape[0] != 0:
            loss += criterion(output01, target01)
        targetCount += target01.shape[0]
        correctNum += calculateCorrect(output01, target01)
    if False:
        output02 = output[:, 4:7][target[:, 1] != -64]
        target02 = target[:, 1:2][target[:, 1] != -64].reshape(-1)
        if output02.shape[0] != 0:
            loss += criterion(output02, target02)
        targetCount += target02.shape[0]
        correctNum += calculateCorrect(output02, target02)
    if True:
        output03 = output[:, 7:][target[:, 2] != -64]
        target03 = target[:, 2:][target[:, 2] != -64].reshape(-1)
        if output03.shape[0] != 0:
            loss += criterion(output03, target03)
        targetCount += target03.shape[0]
        correctNum += calculateCorrect(output03, target03)

    acc = correctNum / (targetCount + 1e-16)

    return loss, acc, targetCount


def train(train_loader, model, criterion, optimizer, epoch, log, tf_writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    # switch to train mode
    model.train()

    end = time.time()
    for i in range(args.iter):
        input, target = train_loader.getTrainInput(args.group, args.tryCount)
        # 根据需要的情况取需要输入的数据

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss, acc, targetCount = calculateLossACC(output, target_var, criterion)

        # measure accuracy and record loss
        losses.update(loss.item(), targetCount)
        top1.update(acc.item(), targetCount)

        # compute gradient and do SGD step
        loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)

        optimizer.step()
        optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if True:
            output = (
                "Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Prec@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                    epoch,
                    i,
                    args.iter,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                    lr=optimizer.param_groups[-1]["lr"] * 0.1,
                )
            )  # TODO
            print(output)
            log.write(output + "\n")
            log.flush()

    tf_writer.add_scalar("loss/train", losses.avg, epoch)
    tf_writer.add_scalar("acc/train_top1", top1.avg, epoch)
    tf_writer.add_scalar("lr", optimizer.param_groups[-1]["lr"], epoch)


def validate(val_loader, model, criterion, epoch, log=None, tf_writer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    calLabel = [[], [], []]
    calResult = [[], [], []]
    calConf = [[], [], []]
    with torch.no_grad():
        for i, (input, target, imagesName) in enumerate(val_loader):
            target = target.cuda()

            # compute output
            output = model(input)

            # measure accuracy and record loss
            loss, acc, targetCount = calculateLossACC(output, target, criterion)
            if targetCount == 0:
                continue

            # -------------------------------------------------------------------
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
                targetT = targetT[targetT != -64]
                if targetT.shape[0] == 0:
                    continue
                (_, _), targetTemp, resultTemp, confTemp = accuracy(
                    outputT.data, targetT, topk=(1, 3)
                )
                calLabel[num].append(targetTemp.cpu())
                calResult[num].append(resultTemp.cpu())
                calConf[num].append(confTemp.cpu())
            # -------------------------------------------------------------------

            losses.update(loss.item(), targetCount)
            top1.update(acc.item(), targetCount)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if True:
                output = (
                    "Test: [{0}/{1}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Prec@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                        16 * i,
                        len(val_loader),
                        batch_time=batch_time,
                        loss=losses,
                        top1=top1,
                    )
                )
                print(output)
                if log is not None:
                    log.write(output + "\n")
                    log.flush()

    aps = []
    for i in range(len(calResult)):
        if len(calLabel[i]) == 0:
            continue
        labels = torch.cat(calLabel[i], 0).cpu().detach()
        results = torch.cat(calResult[i], 0).cpu().detach()
        conf = torch.cat(calConf[i], 0).cpu().detach()
        tp = (labels == results).long()
        apTemp = getConfusionMatrix(
            tp.numpy(), labels.numpy(), results.numpy(), conf.numpy()
        )
        aps.extend(apTemp)

    # 需要参考CVS哪些规则的的测试结果就取哪一部分
    useAps = aps[-4:]
    mAP = sum(useAps) / len(useAps)
    output = "Testing Results: Prec@1 {top1.avg:.3f} Loss {loss.avg:.5f} mAP {mAP:.5f}".format(
        top1=top1, loss=losses, mAP=mAP
    )
    print(output)
    if log is not None:
        log.write(output + "\n")
        log.flush()

    return useAps


# def save_checkpoint(state, model, is_best, filePath, modelPath):
#     torch.save(state, filePath)
#     torch.save(model, modelPath)
#     if is_best:
#         shutil.copyfile(filePath, filePath.replace('pth.tar', 'best.pth.tar'))
#         shutil.copyfile(modelPath, modelPath.replace('.pth', 'Best.pth'))
def save_checkpoint(state, model, filePath, modelPath):
    torch.save(state, filePath)
    torch.save(model, modelPath)


def adjust_learning_rate(optimizer, epoch, lr_type, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if lr_type == "step":
        decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
        lr = args.lr * decay
        decay = args.weight_decay
    elif lr_type == "cos":
        import math

        lr = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.epochs))
        decay = args.weight_decay
    else:
        raise NotImplementedError
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr * param_group["lr_mult"]
        param_group["weight_decay"] = decay * param_group["decay_mult"]


def check_rootfolders():
    """Create log and model folder"""
    folders_util = [
        args.root_log,
        args.root_model,
        os.path.join(args.root_log, args.store_name),
        os.path.join(args.root_model, args.store_name),
    ]
    for folder in folders_util:
        if not os.path.exists(folder):
            print("creating folder " + folder)
            os.mkdir(folder)


def getConfusionMatrix(tp, labels, results, conf):
    p, r, ap, f1, classes = ap_per_class(tp, conf, results, labels)
    print(f"{classes.tolist()}\n{p.tolist()}\n{r.tolist()}\n{ap.tolist()}\n")
    return ap.tolist()


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


if __name__ == "__main__":
    gpuid = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpuid
    main()
