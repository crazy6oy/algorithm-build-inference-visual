# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import os
import time
import json
import shutil
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_

from ops.dataset import TSNDataSet, customCVSDataSet
from ops.models import TSN
from ops.transforms import *
from opts import parser
from ops import dataset_config
from ops.utils import AverageMeter, accuracy
from ops.temporal_shift import make_temporal_pool

from tensorboardX import SummaryWriter
from calculateAP import ap_per_class

best_prec1 = 0


def run_tsm_train(image_folder, label_path, rule, save_folder):
    global args, best_prec1
    args = parser.parse_args()

    with open(label_path, "r", encoding="utf-8") as f:
        labels_msg = json.load(f)

    # 外部传入数据
    image_data_folder = image_folder
    sequence_label_path = label_path
    categories = labels_msg["categories"][rule]
    num_class = len(categories)
    args.root_log = os.path.join(save_folder, "log")
    args.root_model = os.path.join(save_folder, "checkpoints")
    # -----------------------------------------------------------------------------

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

    # save_path = r"D:\work\dataSet\LPDPhase\recognizePhase\9\v4\test.pth"
    # filename = r"D:\work\dataSet\LPDPhase\recognizePhase\9\v4\test2.pth"
    # torch.save(model, save_path)
    # torch.save(model, "test.pth")
    # torch.save(model.state_dict(), filename)

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    # optimizer = torch.optim.SGD(policies,
    #                             args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(policies)

    if args.resume:
        if args.temporal_pool:  # early temporal pool so that we can load the state_dict
            make_temporal_pool(model.module.base_model, args.num_segments)
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint["epoch"]
            best_prec1 = checkpoint["best_prec1"]
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

    train_transform = torchvision.transforms.Compose(
        [
            GroupPad(),
            GroupScale(int(scale_size)),
            GroupCenterCrop(crop_size),
            Stack(roll=(args.arch in ["BNInception", "InceptionV3"])),
            ToTorchFormatTensor(div=(args.arch not in ["BNInception", "InceptionV3"])),
            normalize,
        ]
    )

    valid_transform = torchvision.transforms.Compose(
        [
            GroupPad(),
            GroupScale(int(scale_size)),
            GroupCenterCrop(crop_size),
            Stack(roll=(args.arch in ["BNInception", "InceptionV3"])),
            ToTorchFormatTensor(div=(args.arch not in ["BNInception", "InceptionV3"])),
            normalize,
        ]
    )

    train_loader = torch.utils.data.DataLoader(
        customCVSDataSet(
            image_data_folder,
            sequence_label_path,
            "train",
            rule,
            train_transform,
            args.num_segments,
            categories,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        customCVSDataSet(
            image_data_folder,
            sequence_label_path,
            "valid",
            rule,
            valid_transform,
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
        test_loader = torch.utils.data.DataLoader(
            customCVSDataSet(
                image_data_folder,
                sequence_label_path,
                "test",
                rule,
                valid_transform,
                args.num_segments,
                categories,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )

        validate(test_loader, model, criterion, 0)
        return

    log_training = open(os.path.join(args.root_log, args.store_name, "log.csv"), "w")
    with open(os.path.join(args.root_log, args.store_name, "args.txt"), "w") as f:
        f.write(str(args))
    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_type, args.lr_steps)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, log_training, tf_writer)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch, log_training, tf_writer)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        tf_writer.add_scalar("acc/test_top1_best", best_prec1, epoch)

        output_best = "Best Prec@1: %.3f\n" % (best_prec1)
        print(output_best)
        log_training.write(output_best + "\n")
        log_training.flush()

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "arch": args.arch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_prec1": best_prec1,
            },
            is_best,
        )


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

    output = output[target != -64]
    target = target[target != -64].reshape(-1)
    if output.shape[0] != 0:
        loss += criterion(output, target)
    targetCount += target.shape[0]
    correctNum += calculateCorrect(output, target)

    acc = correctNum / (targetCount + 1e-16)

    return loss, acc, targetCount


def getConfusionMatrix(tp, labels, results, conf):
    p, r, ap, f1, classes = ap_per_class(tp, conf, results, labels)
    print(f"{classes.tolist()}\n{p.tolist()}\n{r.tolist()}\n{ap.tolist()}\n")
    return ap.tolist()


def train(train_loader, model, criterion, optimizer, epoch, log, tf_writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        print(torch.argmax(output, 1).tolist())
        print(target_var.tolist())

        # measure accuracy and record loss
        (prec1, prec5), _, _, _ = accuracy(output.data, target, topk=(1, 3))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)

        optimizer.step()
        optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = (
                "Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                "Prec@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                    epoch,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                    top5=top5,
                    lr=optimizer.param_groups[-1]["lr"],
                )
            )  # TODO
            print(output)
            log.write(output + "\n")
            log.flush()

    tf_writer.add_scalar("loss/train", losses.avg, epoch)
    tf_writer.add_scalar("acc/train_top1", top1.avg, epoch)
    tf_writer.add_scalar("acc/train_top5", top5.avg, epoch)
    tf_writer.add_scalar("lr", optimizer.param_groups[-1]["lr"], epoch)


def validate(val_loader, model, criterion, epoch, log=None, tf_writer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    calLabel = []
    calResult = []
    calConf = []
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()

            # compute output
            output = model(input)

            loss, acc, targetCount = calculateLossACC(output, target, criterion)
            if targetCount == 0:
                continue

            output = output[target != -64]
            target = target[target != -64]
            if target.shape[0] == 0:
                continue
            (_, _), targetTemp, resultTemp, confTemp = accuracy(
                output.data, target, topk=(1, 3)
            )
            calLabel.append(targetTemp.cpu())
            calResult.append(resultTemp.cpu())
            calConf.append(confTemp.cpu())

            losses.update(loss.item(), targetCount)
            top1.update(acc.item(), targetCount)
            batch_time.update(time.time() - end)
            end = time.time()

            output = (
                "Test: [{0}/{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Prec@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                    args.batch_size * i,
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
    if len(calLabel) != 0:
        labels = torch.cat(calLabel, 0).cpu().detach()
        results = torch.cat(calResult, 0).cpu().detach()
        conf = torch.cat(calConf, 0).cpu().detach()
        tp = (labels == results).long()
        apTemp = getConfusionMatrix(
            tp.numpy(), labels.numpy(), results.numpy(), conf.numpy()
        )
        aps.extend(apTemp)

    # 需要参考CVS哪些规则的的测试结果就取哪一部分
    useAps = aps[:]
    mAP = sum(useAps) / len(useAps)
    output = "Testing Results: Prec@1 {top1.avg:.3f} Loss {loss.avg:.5f} mAP {mAP:.5f}".format(
        top1=top1, loss=losses, mAP=mAP
    )
    print(output)
    if log is not None:
        log.write(output + "\n")
        log.flush()

    return mAP


def save_checkpoint(state, is_best):
    filename = "%s/%s/ckpt.pth.tar" % (args.root_model, args.store_name)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace("pth.tar", "best.pth.tar"))


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
