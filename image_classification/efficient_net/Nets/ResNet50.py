import torch
from torch import nn
from torchvision import models


class ResNet50(nn.Module):
    def __init__(self, numClasses):
        super(ResNet50, self).__init__()

        self.resnet50 = models.resnet50(pretrained=False)
        # 加载权重
        pathWeight = "weights/resnet50-19c8e357.pth"
        self.resnet50.load_state_dict(torch.load(pathWeight))

        fcInChannel = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Sequential(
            nn.Linear(fcInChannel, 512), nn.PReLU(), nn.Linear(512, numClasses)
        )

    def forward(self, x):
        x = self.resnet50(x)

        return x


if __name__ == "__main__":
    aIn = torch.randn((2, 3, 416, 416))
    net = ResNet50(3)
    out = net(aIn)
    a = 1
