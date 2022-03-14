import torch
from collections import OrderedDict
from modeling.deeplab import *

model_ft = DeepLab(
    num_classes=6, backbone="resnet", output_stride=16, sync_bn=None, freeze_bn=False
)

model = torch.load(
    r"D:\work\dataSet\organ-segmentation\v2\deepLabv3Plus08\run\custom\deeplab-resnet\organ_segmentation_v8.pth"
)
new_state_dict = OrderedDict()
for k, v in model.state_dict().items():
    name = k[7:]
    new_state_dict[name] = v
model_ft.load_state_dict(new_state_dict)
model_ft.cuda()
a = torch.randn((2, 3, 512, 512)).cuda()
b = model_ft(a)
torch.save(model_ft, "../../organ_segmentation_test.pth")
stop = 0
