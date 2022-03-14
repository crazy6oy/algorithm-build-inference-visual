from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from torchvision import transforms
from dataloaders import custom_transforms as tr


class Trainer(object):
    def __init__(self, nclass, checkpoint_path):
        # Define network
        model = DeepLab(
            num_classes=nclass,
            backbone="resnet",
            output_stride=16,
            sync_bn=False,
            freeze_bn=False,
        )

        # Define Criterion
        self.model = model

        # Using cuda
        self.model = torch.nn.DataParallel(self.model)
        patch_replication_callback(self.model)
        self.model = self.model.cuda()
        self.model.state_dict()

        checkpoint = torch.load(checkpoint_path)
        self.model.module.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        # torch.save(self.model.module, r"D:\work\dataSet\CVS\score\v4\organ_segmentation.pth")
        print("model init final!")

    def runthis(self, img):
        # sample = {'image': img, 'label': mask}
        sample = {"image": img, "label": img}
        composed_transforms = transforms.Compose(
            [tr.Pad(crop_size=512), tr.Normalize(), tr.ToTensor()]
        )

        image = composed_transforms(sample)["image"]
        image = torch.unsqueeze(image, 0).cuda()
        output = self.model(image)
        pred = output.data.cpu().numpy()
        # pred = np.argmax(pred, axis=1)[0]
        return pred


if __name__ == "__main__":
    from PIL import Image, ImageOps
    import numpy as np
    import cv2

    COLORS = [
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
    ]

    LIST_CATEGORY = [
        "background",
        "cystic artery",
        "cystic duct",
        "cystic plate",
        "dissected windows in the hepatocystic triangle",
        "gallbladder",
    ]
    colors = np.array(COLORS, dtype=np.uint)

    # 做图例
    length = 16
    ss = np.ones((len(COLORS) * length, length * 24), dtype=np.int32)
    for i in range(len(COLORS)):
        ss[i * length : (i + 1) * length] *= i
    ss = (
        colors[ss.reshape(-1)]
        .reshape(len(COLORS) * length, length * 24, 3)
        .astype(np.uint8)
    )
    for i in range(len(COLORS)):
        cv2.putText(
            ss,
            LIST_CATEGORY[i],
            (0, length * i + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255 - colors[i]).tolist(),
        )
    # ------------------------------------------------------------------------------------------

    checkpoint_path = r"D:\work\dataSet\CVS\score\v4\model_best.pth.tar"
    video_folder = r"D:\tmp\guangan340.mp4"
    save_folder = r"D:\tmp\save"

    segmentation_inference = Trainer(6, checkpoint_path)
    reader = cv2.VideoCapture(video_folder)
    # 视频writer
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    save_fps = 30
    width = 854
    height = 480
    writer = cv2.VideoWriter(
        r"D:\tmp\guangan340_results.mp4", fourcc, save_fps, (width, height)
    )
    # ------------------------------------------------------------------------------------------
    cc = 0
    while True:
        ret, frame = reader.read()
        if not ret:
            break

        h, w, _ = frame.shape
        image = Image.fromarray(frame)

        probability_matrix = segmentation_inference.runthis(image)
        mask = np.argmax(probability_matrix, axis=1)[0]
        color_map = colors[mask.reshape(-1)].reshape(512, 512, 3).astype(np.uint8)
        color_map = cv2.resize(color_map, (max(h, w), max(h, w)))
        color_map = color_map[:h, :w]
        mask = np.expand_dims(np.sum(color_map, 2) > 0, 2).repeat(3, 2)
        frame[mask] = frame[mask] * 0.4 + color_map[mask] * 0.6

        frame[: ss.shape[0], -ss.shape[1] :] = (
            ss * 0.6 + frame[: ss.shape[0], -ss.shape[1] :] * 0.4
        )

        cv2.putText(
            frame, f"{cc}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (216, 216, 216), 1
        )

        writer.write(frame)
        print(f"{int(cc / 30)}/{22 * 60 + 28}")
        cc += 1
    writer.release()

    # list_image_name = os.listdir(image_folder)
    # for image_name in tqdm(list_image_name):
    #     image_path = os.path.join(image_folder, image_name)
    #     image = Image.open(image_path)
    #     mask = organ_seg.runthis(image)
    #     h, w = mask.shape
    #     colors = np.array(COLORS, dtype=np.uint)
    #     mask = colors[mask.reshape(-1)].reshape(h, w, 3).astype(np.uint8)
    #     # mask = np.expand_dims(mask, 2).repeat(3, 2).astype(np.uint8)
    #     cv2.imwrite(os.path.join(save_folder, image_name.replace(".jpg", ".png")), mask)
