import glob
import os.path

import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import os
import glob
from model import get_timm_model_regression

import yaml
from PIL import Image
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.models import alexnet
import albumentations as A
from pytorch_grad_cam.utils.image import scale_cam_image

combo_val = A.Compose([
    A.Resize(height=224, width=224)
])

import numpy as np
from typing import Callable, List, Optional, Tuple
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection


class NewGradCAM(BaseCAM):
    def __init__(self, model, target_layers,
                 reshape_transform=None):
        super(
            NewGradCAM,
            self).__init__(
            model,
            target_layers,
            reshape_transform)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        # 2D image
        if len(grads.shape) == 4:
            return np.mean(grads, axis=(2, 3))

        # 3D image
        elif len(grads.shape) == 5:
            return np.mean(grads, axis=(2, 3, 4))

        else:
            raise ValueError("Invalid grads shape."
                             "Shape of grads should be 4 (2D image) or 5 (3D image).")

    def get_cam_image(
            self,
            input_tensor: torch.Tensor,
            target_layer: torch.nn.Module,
            targets: List[torch.nn.Module],
            activations: torch.Tensor,
            grads: torch.Tensor,
            eigen_smooth: bool = False,
    ) -> np.ndarray:
        weights = self.get_cam_weights(input_tensor, target_layer, targets, activations, grads)
        # 2D conv
        if len(activations.shape) == 4:
            weighted_activations = weights[:, :, None, None] * activations
        # 3D conv
        elif len(activations.shape) == 5:
            weighted_activations = weights[:, :, None, None, None] * activations
        else:
            raise ValueError(f"Invalid activation shape. Get {len(activations.shape)}.")

        # (n, c, h, w)
        return weighted_activations

    def compute_cam_per_layer(
            self, input_tensor: torch.Tensor, targets: List[torch.nn.Module], eigen_smooth: bool
    ) -> np.ndarray:
        activations_list = [a.cpu().data.numpy() for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy() for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)

        num_sample = len(targets)
        results = []
        # Loop over the saliency image from every layer
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]

            cam = self.get_cam_image(input_tensor, target_layer, targets, layer_activations, layer_grads, eigen_smooth)
            C = cam.shape[1]
            for ci in range(C):
                scaled = scale_cam_image(cam[:, ci], target_size)
                results.append(scaled)
                # cam_per_target_layer.append(scaled[:, None, :])
        return results

    def forward(
            self, input_tensor: torch.Tensor, targets: List[torch.nn.Module], eigen_smooth: bool = False
    ) -> np.ndarray:
        input_tensor = input_tensor.to(self.device)

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True)

        self.outputs = outputs = self.activations_and_grads(input_tensor)
        if targets is None:
            target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
            targets = [ClassifierOutputTarget(category) for category in target_categories]

        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum([target(output) for target, output in zip(targets, outputs)])
            loss.backward(retain_graph=True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(input_tensor, targets, eigen_smooth)
        return cam_per_layer


def get_timm_model(config, pretrained=True):
    import timm
    if config['model'] == 'alexnet':
        model = alexnet(pretrained=pretrained, num_classes=config['num_classes'])
    else:
        model = timm.create_model(config['model'], pretrained=pretrained, num_classes=config['num_classes'])

    return model


norm_trans = Compose([
    ToTensor(),
    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])  # R


def create_input_tensor(path):
    image = Image.open(path).convert('RGB')
    image = np.array(image)
    image = combo_val(image=image)['image']
    tensor = norm_trans(image).unsqueeze(0)
    return tensor


def prepare_image(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (224, 224))
    return image.astype(np.float32) / 255


def draw(ckpt_root, save_dir, test_image_paths=None, gt_classes=None):
    config_path = glob.glob(os.path.join(ckpt_root, "*.yaml"))[0]
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        f.close()
    if test_image_paths is None:
        val_path = config['val_txt']
        with open(val_path, 'r') as f:
            lines = f.readlines()
            f.close()
        import random
        random.shuffle(lines)
        test_image_paths = []
        gt_classes = []
        for i in range(10):
            line = lines[i].strip()
            path = line.split(' ')[0]
            gt_cls = line.split(' ')[-1]
            test_image_paths.append(os.path.join(config['data_root'], path))
            gt_classes.append(int(gt_cls))

    model = get_timm_model_regression(config, pretrained=False)

    ckpt_path = os.path.join(ckpt_root, 'best.pth')
    assert os.path.exists(ckpt_path)
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))

    # print(model)
    #
    # exit(0)
    # TODO
    if config["model"] == 'resnet50':
        # target_layers = [model.act1]
        # target_layers = [model.layer1[-1]]
        # target_layers = [model.layer2[-1]]
        # target_layers = [model.layer3[-1]]
        target_layers = [model.layer4[-1]]
    elif config["model"] == 'alexnet':
        # target_layers = [model.features[1]]
        # target_layers = [model.features[4]]
        # target_layers = [model.features[7]]
        # target_layers = [model.features[9]]
        target_layers = [model.features[11]]
    elif config["model"] == 'densenet169':
        # target_layers = [model.features.norm0.act]
        # target_layers = [model.features.denseblock1.denselayer6.conv2]
        # target_layers = [model.features.denseblock1.denselayer5.conv2]
        # target_layers = [model.features.denseblock1.denselayer4.conv2]
        # target_layers = [model.features.denseblock1.denselayer3.conv2]
        # target_layers = [model.features.denseblock1.denselayer2.conv2]
        target_layers = [model.features.denseblock1.denselayer1.conv2]
    else:
        raise NotImplementedError()

    if isinstance(test_image_paths, str):
        input_tensor = create_input_tensor(test_image_paths)
        bgr_images = [prepare_image(test_image_paths)]
        test_image_paths = [test_image_paths]
    elif isinstance(test_image_paths, (list, tuple)):
        input_tensor = torch.cat([create_input_tensor(x) for x in test_image_paths], dim=0)
        bgr_images = [prepare_image(x) for x in test_image_paths]
    else:
        raise ValueError()

    # We have to specify the target we want to generate the CAM for.
    # if isinstance(gt_classes, int):
    #     targets = [ClassifierOutputTarget(gt_classes)]
    # elif isinstance(gt_classes, (list, tuple)):
    #     targets = [ClassifierOutputTarget(gt) for gt in gt_classes]
    # else:
    #     raise ValueError()

    # Construct the CAM object once, and then re-use it on many images.
    with GradCAM(model=model, target_layers=target_layers) as cam:
        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        grayscale_cams = cam(input_tensor=input_tensor, targets=None, eigen_smooth=True)
        if isinstance(grayscale_cams, list):
            num_channels = len(grayscale_cams)
        else:
            num_channels = None
        # In this example grayscale_cam has only one image in the batch:
        os.makedirs(save_dir, exist_ok=True)

        for i in range(len(bgr_images)):
            fname = os.path.basename(test_image_paths[i])
            fname = os.path.splitext(fname)[0]
            if num_channels is None:
                print(grayscale_cams.shape)
                grayscale_cam = grayscale_cams[i, :]
                # print(grayscale_cam.shape)
                visualization = show_cam_on_image(bgr_images[i], grayscale_cam, image_weight=0, use_rgb=False)
                # You can also get the model outputs without having to redo inference
                # model_outputs = cam.outputs
                bgr = (bgr_images[i] * 255).astype(np.uint8)
                # cv2.imwrite(os.path.join(save_dir, fname), cv2.hconcat([bgr, visualization]))
                cv2.imwrite(os.path.join(save_dir, fname + ".png"), visualization)
            else:
                for ci in range(num_channels):
                    grayscale_cam = grayscale_cams[ci][i, :]
                    # print(grayscale_cam.shape)
                    visualization = show_cam_on_image(bgr_images[i], grayscale_cam, image_weight=0, use_rgb=False)
                    # You can also get the model outputs without having to redo inference
                    # model_outputs = cam.outputs

                    bgr = (bgr_images[i] * 255).astype(np.uint8)
                    cc = str(ci).zfill(3)
                    # cv2.imwrite(os.path.join(save_dir, fname), cv2.hconcat([bgr, visualization]))
                    cv2.imwrite(os.path.join(save_dir, fname + f"_c{cc}.png"), visualization)


if __name__ == "__main__":

    # tag = "exp2_depth_alexnet"
    # tag = "exp4_pose_alexnet"
    # tag = "exp5_pose_resnet50"
    # tag = "exp6_pose_densenet169"
    # tag = "exp9_depth_regression_resnet50"
    # tag = "exp10_depth_regression_alexnet"
    tag = 'exp11_depth_regression_densenet169'

    model = tag.split('_')[-1]
    # paths = []
    # labels = []
    # for file in sorted(os.listdir("0_0/cropped_images")):
    #     paths.append(os.path.join("0_0/cropped_images", file))
    #     labels.append(0)

    # split = [1, 2, 3, 4]
    # split = [0] + split + [len(paths)]
    # split = [0, 1, 2, 3, 4]

    # for i in range(len(split) - 1):
    #     ckpt_root = f"results/{tag}"
    #     save_dir = f"export_results/act1_w0"
    #     draw(ckpt_root, save_dir, test_image_paths=paths[split[i]:split[i + 1]], gt_classes=labels)

    ckpt_root = f"results/{tag}"
    save_dir = f"export_results4/{model}_denseblock1_1_0_0"
    path = "0.png"
    draw(ckpt_root, save_dir, test_image_paths=[path], gt_classes=None)
    save_dir = f"export_results4/{model}_denseblock1_1_70_70"
    path = "70_70.png"
    draw(ckpt_root, save_dir, test_image_paths=[path], gt_classes=None)
