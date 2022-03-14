import sys

sys.path.append("./image_classification/efficient_net")

# Boilerplate imports.
import numpy as np
import PIL.Image
from matplotlib import pylab as P
import torch
from torchvision import models, transforms

import saliency.core as saliency
from image_classification.efficient_net.demo import EfficientNetClassification


class alg(object):
    def __init__(self):
        model = models.inception_v3(pretrained=True, init_weights=False)
        self.model = model.cuda()
        self.eval_mode = model.eval()

    def __format__(self, img):
        im_tensor = self.preprocess_images(img)
        im_tensor = im_tensor.cuda()
        predictions = self.model(im_tensor)
        return predictions

    def __call__(self, img):
        return self.__format__(img)

    def preprocess_images(self, images):
        # assumes input is 4-D, with range [0,255]
        #
        # torchvision have color channel as first dimension
        # with normalization relative to mean/std of ImageNet:
        #    https://pytorch.org/vision/stable/models.html

        images = images.numpy()

        transformer = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        images = np.array(images)
        images = images / 255
        images = np.transpose(images, (0, 3, 1, 2))
        images = torch.tensor(images, dtype=torch.float32)
        images = transformer.forward(images)
        return images.requires_grad_(True)

    def inference(self, input):
        im_tensor = input.cuda()
        predictions = self.model(im_tensor)
        return predictions


BATCH_SIZE = 2


class saliency_visualization(object):
    def __init__(self, alg_inference):
        self.alg_inference = alg_inference
        self.model_train = alg_inference.model.train()
        self.model_eval = alg_inference.model.eval()
        # Register hooks for Grad-CAM, which uses the last convolution layer
        # conv_layer = self.model_train.Mixed_7c
        conv_layer = self.model_train._conv_head
        self.conv_layer_outputs = {}

        conv_layer.register_forward_hook(self.conv_layer_forward)
        conv_layer.register_full_backward_hook(self.conv_layer_backward)

    def conv_layer_forward(self, m, i, o):
        # move the RGB dimension to the last dimension
        self.conv_layer_outputs[saliency.base.CONVOLUTION_LAYER_VALUES] = (
            torch.movedim(o, 1, 3).detach().cpu().numpy()
        )

    def conv_layer_backward(self, m, i, o):
        # move the RGB dimension to the last dimension
        self.conv_layer_outputs[saliency.base.CONVOLUTION_OUTPUT_GRADIENTS] = (
            torch.movedim(o[0], 1, 3).detach().cpu().numpy()
        )

    def call_model_function(self, images, call_model_args=None, expected_keys=None):
        target_class_idx = call_model_args
        # images = [PIL.Image.fromarray(images[i].astype(np.uint8)) for i in range(images.shape[0])]
        images = torch.from_numpy(images)
        # for x in images:
        #     x.show()

        images = self.alg_inference.preprocess_images(images)
        # images = self.alg_inference.preprocess_images_with_segmentation(images)
        output = self.alg_inference.calculate(images)
        m = torch.nn.Softmax(dim=1)
        output = m(output)
        self.result_id = torch.argmax(torch.bincount(torch.argmax(output, 1)), 0)
        if saliency.base.INPUT_OUTPUT_GRADIENTS in expected_keys:
            outputs = output[:, target_class_idx]
            grads = torch.autograd.grad(
                outputs, images, grad_outputs=torch.ones_like(outputs)
            )
            grads = torch.movedim(grads[0], 1, 3)
            gradients = grads.detach().numpy()
            return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}
        else:
            one_hot = torch.zeros_like(output)
            one_hot[:, target_class_idx] = 1
            self.model_train.zero_grad()
            output.backward(gradient=one_hot, retain_graph=True)
            return self.conv_layer_outputs

    def vg_s(self, im_orig, target_id):
        # Vanilla Gradient & SmoothGrad
        # Construct the saliency object. This alone doesn't do anthing.
        integrated_gradients = saliency.IntegratedGradients()

        # Baseline is a black image.
        im = im_orig.astype(np.float32)
        baseline = np.zeros(im.shape)

        # Compute the vanilla mask and the smoothed mask.
        vanilla_integrated_gradients_mask_3d = integrated_gradients.GetMask(
            im,
            self.call_model_function,
            target_id,
            x_steps=25,
            x_baseline=baseline,
            batch_size=BATCH_SIZE,
        )
        # Smoothed mask for integrated gradients will take a while since we are doing nsamples * nsamples computations.
        smoothgrad_integrated_gradients_mask_3d = integrated_gradients.GetSmoothedMask(
            im,
            self.call_model_function,
            target_id,
            x_steps=25,
            x_baseline=baseline,
            batch_size=BATCH_SIZE,
        )

        # Call the visualization methods to convert the 3D tensors to 2D grayscale.
        vanilla_mask_grayscale = saliency.VisualizeImageGrayscale(
            vanilla_integrated_gradients_mask_3d
        )
        smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(
            smoothgrad_integrated_gradients_mask_3d
        )

        P.close()
        # Set up matplot lib figures.
        ROWS = 1
        COLS = 3
        UPSCALE_FACTOR = 20
        P.figure(figsize=(COLS * UPSCALE_FACTOR, ROWS * UPSCALE_FACTOR))

        # Render the saliency masks.
        ShowImage(im_orig, title="Original Image", ax=P.subplot(ROWS, COLS, 1))
        ShowGrayscaleImage(
            vanilla_mask_grayscale,
            title="Vanilla Integrated Gradients",
            ax=P.subplot(ROWS, COLS, 2),
        )
        ShowGrayscaleImage(
            smoothgrad_mask_grayscale,
            title="Smoothgrad Integrated Gradients",
            ax=P.subplot(ROWS, COLS, 3),
        )
        # P.show()

    def ig_s(self):
        # Integrated Gradients & SmoothGrad
        pass

    def xrai(self, im_orig, target_id, top_threshold=30):
        # XRAI
        xrai_object = saliency.XRAI()

        im = im_orig.astype(np.float32)
        xrai_attributions = xrai_object.GetMask(
            im, self.call_model_function, target_id, batch_size=BATCH_SIZE
        )

        P.close()
        # Set up matplot lib figures.
        ROWS = 1
        COLS = 3
        UPSCALE_FACTOR = 20
        P.figure(figsize=(COLS * UPSCALE_FACTOR, ROWS * UPSCALE_FACTOR))

        # Show original image
        ShowImage(im_orig, title="Original Image", ax=P.subplot(ROWS, COLS, 1))

        # Show XRAI heatmap attributions
        ShowHeatMap(
            xrai_attributions, title="XRAI Heatmap", ax=P.subplot(ROWS, COLS, 2)
        )

        # Show most salient 30% of the image
        mask = xrai_attributions > np.percentile(xrai_attributions, 100 - top_threshold)
        im_mask = np.array(im_orig)
        im_mask[~mask] = 0
        ShowImage(im_mask, title="Top 30%", ax=P.subplot(ROWS, COLS, 3))
        # P.show()

    def xrai_fast(self):
        # XRAI Fast
        pass

    def gc(self, im_orig, target_id):
        # Grad-CAM
        # Compare Grad-CAM and Smoothgrad with Grad-CAM.

        # Construct the saliency object. This alone doesn't do anthing.
        grad_cam = saliency.GradCam()

        # Compute the Grad-CAM mask and Smoothgrad+Grad-CAM mask.
        im = im_orig.astype(np.float32)
        grad_cam_mask_3d = grad_cam.GetMask(im, self.call_model_function, target_id)
        smooth_grad_cam_mask_3d = grad_cam.GetSmoothedMask(
            im, self.call_model_function, target_id
        )

        # Call the visualization methods to convert the 3D tensors to 2D grayscale.
        grad_cam_mask_grayscale = saliency.VisualizeImageGrayscale(grad_cam_mask_3d)
        smooth_grad_cam_mask_grayscale = saliency.VisualizeImageGrayscale(
            smooth_grad_cam_mask_3d
        )

        P.close()
        # Set up matplot lib figures.
        ROWS = 1
        COLS = 3
        UPSCALE_FACTOR = 20
        P.figure(figsize=(COLS * UPSCALE_FACTOR, ROWS * UPSCALE_FACTOR))

        # Render the saliency masks.
        ShowImage(im_orig, title="Original Image", ax=P.subplot(ROWS, COLS, 1))
        ShowGrayscaleImage(
            grad_cam_mask_grayscale, title="Grad-CAM", ax=P.subplot(ROWS, COLS, 2)
        )
        ShowGrayscaleImage(
            smooth_grad_cam_mask_grayscale,
            title="Smoothgrad Grad-CAM",
            ax=P.subplot(ROWS, COLS, 3),
        )
        # P.show()

    def g_ig(self):
        # Guided IG
        pass

    def b_ig(self):
        # Blur IG
        pass

    def compare_b_big(self):
        # Compare BlurIG and Smoothgrad with BlurIG.
        pass


if __name__ == "__main__":
    import os
    import json
    import shutil
    import tqdm

    def ShowImage(im, title="", ax=None):
        if ax is None:
            P.figure()
        P.axis("off")
        P.imshow(im)
        P.title(title)

    def ShowGrayscaleImage(im, title="", ax=None):
        if ax is None:
            P.figure()
        P.axis("off")
        P.imshow(im, cmap=P.cm.gray, vmin=0, vmax=1)
        P.title(title)

    def ShowHeatMap(im, title, ax=None):
        if ax is None:
            P.figure()
        P.axis("off")
        P.imshow(im, cmap="inferno")
        P.title(title)

    def LoadImage(file_path):
        im = PIL.Image.open(file_path)
        im = im.resize((380, 380))
        if im.mode == "RGBA":
            im = np.asarray(im)[..., :3]
        else:
            im = np.asarray(im)
        return im

    image_folder = r"D:\work\dataSet\CVS\for sequence\CVSSequenceFrames20210610"
    json_path = r"D:\work\dataSet\CVS\score\v5\cvs_score_for_images_v1.json"
    # model_path = r"D:\work\dataSet\CVS\score\v5\results\efficient_net\2022-3-3_22.18.25\checkpoints\efficientnet-b4_best_mAP.pth"
    model_path = "cvs_score_test_in3.pth"
    # model_path = "cvs_score_test_in9.pth"
    other_model_path = r"D:\work\dataSet\organ-segmentation\v2\deepLabv3Plus08\run\custom\deeplab-resnet\organ_segmentation_v8.pth"
    with open(json_path, encoding="utf-8") as f:
        label_msg = json.load(f)
    alg_infer = EfficientNetClassification(model_path,other_model_path)
    visual = saliency_visualization(alg_infer)

    categories = label_msg["categories"]["cvsiii"]
    for score in label_msg["labels"]["cvsiii"]["test"].keys():
        target_id = categories.index(score)
        # target_id = 236
        for img_name in tqdm.tqdm(label_msg["labels"]["cvsiii"]["test"][score]):
            im_orig = LoadImage(os.path.join(image_folder, img_name))
            # im_orig = LoadImage("./doberman.png")
            # alg_infer = alg()
            visual.vg_s(im_orig, target_id)
            P.show()
            # P.savefig(f"./result/visual/{img_name.split('.')[0]}_vgsg_target_{target_id}_result_{visual.result_id}.png")
            visual.xrai(im_orig, target_id)
            P.show()
            # P.savefig(f"./result/visual/{img_name.split('.')[0]}_xrai_target_{target_id}_result_{visual.result_id}.png")
            visual.gc(im_orig, target_id)
            P.show()
            # P.savefig(f"./result/visual/{img_name.split('.')[0]}_gc_target_{target_id}_result_{visual.result_id}.png")
