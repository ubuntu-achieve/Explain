import os
import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import torch
from torchvision.models import vgg19, resnet50

from lib.gradients import GradCam, GuidedBackpropGrad
from lib.image_utils import preprocess_image
from lib.labels import IMAGENET_LABELS

def main(img, use_cuda, model_name='vgg19'):
    target_index = None
    img = np.float32(cv2.resize(img, (224, 224))) / 255
    # 图像预处理
    preprocessed_img = preprocess_image(img, use_cuda)
    if model_name == 'vgg19':
        model = vgg19(pretrained=True)
        target_layer_names = ['35']
    elif model_name == 'resnet50':
        model = resnet50(pretrained=True)
        target_layer_names = ['layer4']
    model.eval()
    if use_cuda:
        model.cuda()

    # 预测
    output = model(preprocessed_img)
    pred_index = np.argmax(output.data.cpu().numpy())
    print('Prediction: {}'.format(IMAGENET_LABELS[pred_index]))

    # 准备grad cam
    grad_cam = GradCam(
        pretrained_model=model,
        target_layer_names=target_layer_names,
        cuda=use_cuda
    )
    # 计算 grad cam
    mask = grad_cam(preprocessed_img, target_index)

    # 计算 guided backpropagation
    guided_backprop = GuidedBackpropGrad(
        pretrained_model=model,
        cuda=use_cuda
    )
    guided_backprop_saliency = guided_backprop(preprocessed_img, index=target_index)

    cam_mask = np.zeros(guided_backprop_saliency.shape)
    for i in range(guided_backprop_saliency.shape[0]):
        cam_mask[i, :, :] = mask

    cam_guided_backprop = np.multiply(cam_mask, guided_backprop_saliency)
    return cam_guided_backprop

# 优化显示
def enhance(s_mask, s_img, leve=0.025):
    try:
        mask = np.transpose(s_mask, (1,2,0)).copy()
        img  = s_img.copy()
    except:
        mask = s_mask.copy()
        img  = s_img.copy()
    mask[np.where(mask > leve)] = 255
    mask[np.where(mask < leve)] = 0
    img = cv2.resize(mask, img.shape[:-1][::-1])*0.5 + img*0.5
    return img

use_cuda    = False and torch.cuda.is_available()
input_path  = './Images'
output_path = 'Results/Backpropagation'
model_name  = 'vgg19'

if __name__ == "__main__":
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for img in os.listdir(input_path):
        # 读取图片
        image    = cv2.imread(os.path.join(input_path, img))
        # 开始计算
        mask     = main(image.copy(), use_cuda, model_name)
        # 增强显示
        mask_pro = enhance(mask, image, leve=0.05)
        cv2.imwrite(os.path.join(output_path, 'Backpropagation_'+model_name+'_'+img), np.transpose(mask, (1,2,0))*255)
        cv2.imwrite(os.path.join(output_path, 'Backpropagation_enhance_'+model_name+'_'+img), mask_pro)