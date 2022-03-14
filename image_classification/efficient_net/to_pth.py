import torch
from collections import OrderedDict
from efficientnet_pytorch import EfficientNet

model_ft = EfficientNet.from_pretrained("efficientnet-b4", in_channels=9, num_classes=4)

model = torch.load(
    r"D:\work\dataSet\CVS\score\v5\results\efficient_net\2022-3-5_3.6.6\checkpoints\efficientnet-b4_best_mAP.pth"
)
new_state_dict = OrderedDict()
for k, v in model.state_dict().items():
    name = k[7:]
    new_state_dict[name] = v
model_ft.load_state_dict(new_state_dict)
model_ft.cuda()
a = torch.randn((2, 9, 380, 380)).cuda()
b = model_ft(a)
torch.save(model_ft, "../../cvs_score_test_in9.pth")
stop = 0
