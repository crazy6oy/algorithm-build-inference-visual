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

best_prec1 = 0


def mk_onnx(state_dict):
    global args, best_prec1
    args = parser.parse_args()

    dictCategories = {"phase": ["0", "1", "2", "3", "4", "5", "6"]}
    # dictCategories = {'phase': ["-1", "0", "1", "2", "-1", "0", "2", "-1", "0", "1", "2"]}
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

    model.load_state_dict(state_dict)
    stop = 0

    # if args.tune_from:
    #     print(("=> fine-tuning from '{}'".format(args.tune_from)))
    #     sd = torch.load(args.tune_from)
    #     sd = sd['state_dict']
    #     model_dict = model.state_dict()
    #     replace_dict = []
    #     for k, v in sd.items():
    #         if k not in model_dict and k.replace('.net', '') in model_dict:
    #             print('=> Load after remove .net: ', k)
    #             replace_dict.append((k, k.replace('.net', '')))
    #     for k, v in model_dict.items():
    #         if k not in sd and k.replace('.net', '') in sd:
    #             print('=> Load after adding .net: ', k)
    #             replace_dict.append((k.replace('.net', ''), k))
    #
    #     for k, k_new in replace_dict:
    #         sd[k_new] = sd.pop(k)
    #     keys1 = set(list(sd.keys()))
    #     keys2 = set(list(model_dict.keys()))
    #     set_diff = (keys1 - keys2) | (keys2 - keys1)
    #     print('#### Notice: keys that failed to load: {}'.format(set_diff))
    #     if args.dataset not in args.tune_from:  # new dataset
    #         print('=> New dataset, do not load fc weights')
    #         sd = {k: v for k, v in sd.items() if 'fc' not in k}
    #     if args.modality == 'Flow' and 'Flow' not in args.tune_from:
    #         sd = {k: v for k, v in sd.items() if 'conv1.weight' not in k}
    #     model_dict.update(sd)
    #     model.load_state_dict(model_dict)

    if args.temporal_pool and not args.resume:
        make_temporal_pool(model.module.base_model, args.num_segments)
    model.eval()
    # save_path = r"D:\work\dataSet\LCPhase\recognition_lc_phase\v1\3\tsm_lc_phase_05\checkpoint\TSM_something_RGB_resnet50_shift8_blockres_avg_segment8_e128\ckpt.best.pth"
    # torch.save(model, save_path)

    # 1 24 224 224
    example = torch.ones((8, 3, 224, 224))
    # torch.onnx.export(model, example, "LPD1+3V1.onnx")

    trace_path = r"D:\work\dataSet\LCPhase\recognition_lc_phase\v1\3\tsm_lc_phase_05\checkpoint\TSM_something_RGB_resnet50_shift8_blockres_avg_segment8_e128\ckpt.best.pt"
    if os.path.exists(trace_path):
        return
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module.save(trace_path)


def change_feature(check_point):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # 由于本文中是使用cpu，因此使用torch.load中将设备加载到cpu中，实际上可以直接使用torch.load进行加载，默认是cpu设备。
    check_point = torch.load(check_point, map_location=device)["state_dict"]
    return check_point

    import collections

    dicts = collections.OrderedDict()

    for k, value in check_point.items():
        print("names:{}".format(k))  # 打印结构
        print("shape:{}".format(value.size()))

        if "module" in k:  # 去除命名中的module
            k = k.split(".")[1:]
            k = ".".join(k)
            print(k)
        else:
            stop = 0
        dicts[k] = value

    return dicts


if __name__ == "__main__":
    gpuid = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpuid

    model_path = r"D:\work\dataSet\LCPhase\recognition_lc_phase\v1\3\tsm_lc_phase_05\checkpoint\TSM_something_RGB_resnet50_shift8_blockres_avg_segment8_e128\ckpt.best.pth.tar"

    state_dict = change_feature(model_path)
    mk_onnx(state_dict)
