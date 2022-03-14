import torch
from PIL import Image

from torchvision import transforms
from dataloaders import custom_transforms as tr
import tensor_transforms as ttr


class OrganSegmentation(object):
    def __init__(self, model_path):
        self.model = torch.load(model_path)
        self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.cuda()
        self.model.eval()
        print(f"{model_path} is loaded")
        self.input_size = 512
        self.composed_transforms = transforms.Compose(
            [
                tr.Pad(crop_size=self.input_size),
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor(),
            ]
        )
        self.composed_transforms_tensor = transforms.Compose(
            [
                ttr.Permute_Tensor(),
                ttr.Pad_Tensor(),
                ttr.Resize_Tensor(self.input_size),
                ttr.Normalize_Tensor(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

    def inference(self, img):
        # img = img.resize((self.input_size, self.input_size), Image.BILINEAR)
        sample = {"image": img, "label": img}

        image = self.composed_transforms(sample)["image"]
        image = torch.unsqueeze(image, 0).cuda()
        output = self.model(image)
        return output.data

    def preprocess_images(self, image_data) -> torch.Tensor:
        if len(image_data.shape) == 3:
            image_data = torch.unsqueeze(image_data, 0)
        image_data = self.composed_transforms_tensor(image_data)

        return image_data.float().requires_grad_(True)

    def calculate(self, input):
        if input.device.type == "cpu":
            input = input.cuda()
        return self.model(input)


if __name__ == "__main__":
    from PIL import Image
    import cv2
    import numpy as np

    colors = [
        [0, 0, 0],
        [200, 0, 0],
        [0, 200, 0],
        [0, 0, 200],
        [200, 200, 0],
        [200, 0, 200],
        [0, 200, 200],
    ]
    classes = {
        1: "cystic artery",
        2: "cystic duct",
        3: "cystic plate",
        4: "windows",
        5: "gallbladder",
    }

    def draw_example(classes, colors):
        max_id = max(classes.keys())
        max_string = max([len(x) for x in classes.values()])
        height = 16
        width = 16
        char_width = 9.3

        output = np.zeros(
            (max_id * height, width + int(max_string * char_width), 3), dtype=np.uint8
        )

        for class_id in classes.keys():
            output[(class_id - 1) * height + 1 : class_id * height, 1:16] = colors[
                class_id
            ]
            cv2.putText(
                output,
                classes[class_id],
                (1 + width, class_id * 16 - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                colors[class_id],
            )
        return output

    example_img = draw_example(classes, colors)

    model_path = r"D:\work\dataSet\organ-segmentation\v2\deepLabv3Plus08\run\custom\deeplab-resnet\organ_segmentation_v8.pth"
    # model_path = r"D:\work\dataSet\organ-segmentation\v2\deepLabv3Plus07\run\custom\deeplab-xception\model_best.pth"
    video_path = r"D:\tmp\202203041532.mp4"

    model = OrganSegmentation(model_path)
    video_capture = cv2.VideoCapture(video_path)

    colors_np = np.array(colors, dtype=np.uint8)
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        model_result = model.inference(img)[0].cpu().numpy()
        id_mask = np.argmax(model_result, 0)
        color_mask = colors_np[id_mask]
        color_mask = cv2.resize(
            color_mask, (600, 600), interpolation=cv2.INTER_NEAREST
        )[:480]

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        segmentation_visual = (color_mask * 0.4 + frame * 0.6).astype(np.uint8)
        segmentation_visual[
            : example_img.shape[0], -example_img.shape[1] :
        ] = example_img
        cv2.imshow("play", segmentation_visual)
        cv2.waitKey(1)

        stop = 0
