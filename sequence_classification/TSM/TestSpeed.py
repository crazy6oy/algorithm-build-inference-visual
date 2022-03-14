import time
import torch

model_path = (
    r"D:\work\dataSet\LPDPhase\recognizePhase\9\v4（调整了数据划分，包含了体腔外数据）\ckpt24_v2.best.pth"
)
model = torch.load(model_path).cuda()
input = torch.randn((8, 3, 224, 224)).cuda()

while True:
    time1 = time.time()
    _ = model(input)
    time2 = time.time()
    print(f"FPS: {1 / (time2 - time1)}")
