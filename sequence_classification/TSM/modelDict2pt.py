import os
import time
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

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    # num_class, args.train_list, args.val_list, args.root_path, prefix = dataset_config.return_dataset(args.dataset,
    #                                                                                                   args.modality)
    # args.root_path = "something-something"
    categories = ["0", "1", "2", "3", "4"]
    num_class = len(categories)

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

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    modelPath = "/home/withai/wangyx/checkPoint/TSM/TSM_LPDv0.pth.tar"
    checkpoint = torch.load(modelPath)
    args.start_epoch = checkpoint["epoch"]
    best_prec1 = checkpoint["best_prec1"]
    model.load_state_dict(checkpoint["state_dict"])
    torch.save(model, "/home/withai/wangyx/checkPoint/TSM/LPDv0.pth.tar")


if __name__ == "__main__":
    gpuid = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpuid
    main()
