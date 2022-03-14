import sys

sys.path.append("./image_segmentation/deeplab_v3plus")

import os
import torch
from torchvision import transforms
import tensor_transforms as ttr
import image_classification.efficient_net.utils.transforms as tr
from image_segmentation.deeplab_v3plus.demo import OrganSegmentation


class EfficientNetClassification(object):
    def __init__(self, model_path, other_model_path=None):
        self.model = torch.load(model_path)
        self.model.cuda()
        self.model.eval()
        print(f"{model_path} is loaded!")

        if other_model_path and os.path.exists(other_model_path):
            self.other_inference = OrganSegmentation(other_model_path)

        input_size = 380
        self.composed_transforms = transforms.Compose(
            [tr.Pad(input_size), tr.Normalize(), tr.ToTensor()]
        )
        self.composed_transforms_tensor = transforms.Compose(
            [
                ttr.Permute_Tensor(),
                ttr.Pad_Tensor(),
                ttr.Resize_Tensor(input_size),
                ttr.Normalize_Tensor(),
            ]
        )

    def inference(self, image):
        input_data = self.preprocess_images(image)
        return self.calculate(input_data)

    def preprocess_images(self, image_data) -> torch.Tensor:
        if len(image_data.shape) == 3:
            image_data = torch.unsqueeze(image_data, 0)
        image_data = self.composed_transforms_tensor(image_data)

        return image_data.float().requires_grad_(True)

    def calculate(self, input):
        if input.device.type == "cpu":
            input = input.cuda()
        return self.model(input)

    def preprocess_images_with_segmentation(self, image_data) -> torch.Tensor:
        if len(image_data.shape) == 3:
            image_data = torch.unsqueeze(image_data, 0)
        input_data = self.composed_transforms_tensor(image_data)

        other_input_data = self.other_inference.preprocess_images(image_data)
        probability_matrix = self.other_inference.calculate(other_input_data)
        probability_matrix = torch.nn.functional.interpolate(
            probability_matrix,
            (input_data.shape[2], input_data.shape[3]),
            mode="bilinear",
        )
        probability_matrix = probability_matrix.detach().cpu()

        return (
            torch.cat((input_data, probability_matrix), 1).float().requires_grad_(True)
        )

    def inference_with_segmentation(self, image):
        input_data = self.preprocess_images_with_segmentation(image)
        return self.calculate(input_data)
