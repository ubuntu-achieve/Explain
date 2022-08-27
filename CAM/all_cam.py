import os
import cv2
import numpy as np
import torch
from torchvision import models
from pytorch_grad_cam import GradCAM ,GradCAMPlusPlus, GradCAMElementWise, XGradCAM, AblationCAM, ScoreCAM,\
    EigenCAM, EigenGradCAM, LayerCAM, FullGrad, HiResCAM, RandomCAM

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import warnings
warnings.filterwarnings("ignore")

methods = {
    'HiResCAM':HiResCAM,
    'RandomCAM':RandomCAM,
    'Grad-CAM':GradCAM,
    'Grad-CAM++':GradCAMPlusPlus,
    'Grad-CAM-ElementWise':GradCAMElementWise,
    'XGrad-CAM':XGradCAM,
    'Ablation-CAM':AblationCAM,
    'Score-CAM':ScoreCAM,
    'Eigen-CAM':EigenCAM,
    'EigenGrad-CAM':EigenGradCAM,
    'Layer-CAM':LayerCAM,
    'Full-Grad':FullGrad,
}

input_path  = "./Images"
use_cuda = False

if __name__ == '__main__':
    for method in [i for i in methods.keys()][7:]:
        output_path = f'./Results/{method}'
        print(f'{method}测试')
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        for img in os.listdir(input_path):
            # 输入图片
            image_path = os.path.join(input_path, img)
            # 载入 ResNet50图像分类模型和vgg19
            model = [models.vgg19(pretrained=True), models.resnet50(pretrained=True)]
            output_cam, output_gb, output_cam_gb = [], [], []
            for m in model:
                try:
                    target_layers = [m.layer4]
                except:
                    target_layers = [m.features]
                rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
                rgb_img = np.float32(rgb_img) / 255
                input_tensor = preprocess_image(rgb_img,
                                                mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])


                targets = None

                cam_algorithm = methods[method]
                with cam_algorithm(model=m, target_layers=target_layers, use_cuda=use_cuda) as cam:

                    cam.batch_size = 32
                    grayscale_cam = cam(input_tensor=input_tensor,
                                        targets=targets,
                                        aug_smooth=False,
                                        eigen_smooth=False)

                    # Here grayscale_cam has only one image in the batch
                    grayscale_cam = grayscale_cam[0, :]

                    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

                    # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
                    cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

                gb_model = GuidedBackpropReLUModel(model=m, use_cuda=use_cuda)
                gb = gb_model(input_tensor, target_category=None)

                cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
                cam_gb = deprocess_image(cam_mask * gb)
                gb = deprocess_image(gb)
                output_cam.append(cam_image)
                output_gb.append(gb)
                output_cam_gb.append(cam_gb)

            cv2.imwrite(os.path.join(output_path, f'{method}_cam_' + img), np.hstack(output_cam))
            cv2.imwrite(os.path.join(output_path, f'{method}gb_' + img), np.hstack(output_gb))
            cv2.imwrite(os.path.join(output_path, f'{method}cam_gb_' + img), np.hstack(output_cam_gb))