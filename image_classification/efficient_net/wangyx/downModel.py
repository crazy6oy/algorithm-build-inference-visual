from efficientnet_pytorch import EfficientNet

if __name__ == "__main__":

    # train
    pth_map = {
        "efficientnet-b0": "efficientnet-b0-355c32eb.pth",
        "efficientnet-b1": "efficientnet-b1-f1951068.pth",
        "efficientnet-b2": "efficientnet-b2-8bb594d6.pth",
        "efficientnet-b3": "efficientnet-b3-5fb5a3c3.pth",
        "efficientnet-b4": "efficientnet-b4-6ed6700e.pth",
        "efficientnet-b5": "efficientnet-b5-b6417697.pth",
        "efficientnet-b6": "efficientnet-b6-c76e70fd.pth",
        "efficientnet-b7": "efficientnet-b7-dcc49843.pth",
    }
    # 自动下载到本地预训练
    model = EfficientNet.from_pretrained("efficientnet-b5")
