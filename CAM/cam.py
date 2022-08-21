import os
import cv2
import warnings
import numpy as np
from PIL import Image
from torch.nn import functional as F
from torchvision import models, transforms
warnings.filterwarnings("ignore")

# 测试图像地址&&结果保存地质
input_path  = "./Images"
output_path = "./Results/CAM"
if not os.path.isdir(output_path):
    os.makedirs(output_path)

def feature_hook(model, input, output):
    feature_data.append(output.data.numpy())

# =====获取CAM start=====
def makeCAM(feature, weights, classes_id):
    print(feature.shape, weights.shape, classes_id)
    # batchsize, C, h, w
    bz, nc, h, w = feature.shape
    # (512,) @ (512, 7*7) = (49,)
    cam = weights[classes_id].dot(feature.reshape(nc, h * w))
    cam = cam.reshape(h, w)  # (7, 7)
    # 归一化到[0, 1]之间
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    # 转换为0～255的灰度图
    cam_gray = np.uint8(255 * cam)
    # 最后，上采样操作，与网络输入的尺寸一致，并返回
    return cv2.resize(cam_gray, (224, 224))

for img in os.listdir(input_path):
    img_path = os.path.join(input_path, img)

    # 定义预训练模型: resnet18、resnet50、densenet121
    resnet18 = models.resnet18(pretrained=True)
    resnet50 = models.resnet50(pretrained=True)
    densenet121 = models.densenet121(pretrained=True)
    resnet18.eval()
    resnet50.eval()
    densenet121.eval()

    # =====注册hook start=====
    feature_data = []
    resnet18._modules.get('layer4').register_forward_hook(feature_hook)
    resnet50._modules.get('layer4').register_forward_hook(feature_hook)
    densenet121._modules.get('features').register_forward_hook(feature_hook)
    # =====注册hook end=====

    # 获取fc层的权重
    fc_weights_resnet18 = resnet18._modules.get('fc').weight.data.numpy()
    fc_weights_resnet50 = resnet50._modules.get('fc').weight.data.numpy()
    fc_weights_densenet121 = densenet121._modules.get('classifier').weight.data.numpy()

    # 图片数据转换
    image_transform = transforms.Compose([
        # 将输入图片resize成统一尺寸
        transforms.Resize([224, 224]),
        # 将PIL Image或numpy.ndarray转换为tensor，并除255归一化到[0,1]之间
        transforms.ToTensor(),
        # 标准化处理-->转换为标准正太分布，使模型更容易收敛
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    # 获取预测类别id
    image = image_transform(Image.open(img_path)).unsqueeze(0)
    out_resnet18 = resnet18(image)
    out_resnet50 = resnet50(image)
    out_densenet121 = densenet121(image)
    predict_classes_id_resnet18 = np.argmax(F.softmax(out_resnet18, dim=1).data.numpy())
    predict_classes_id_resnet50 = np.argmax(F.softmax(out_resnet50, dim=1).data.numpy())
    predict_classes_id_densenet121 = np.argmax(F.softmax(out_densenet121, dim=1).data.numpy())

    cam_gray_resnet18 = makeCAM(feature_data[0], fc_weights_resnet18, predict_classes_id_resnet18)
    cam_gray_resnet50 = makeCAM(feature_data[1], fc_weights_resnet50, predict_classes_id_resnet50)
    cam_gray_densenet121 = makeCAM(feature_data[2], fc_weights_densenet121, predict_classes_id_densenet121)
    # =====获取CAM start=====

    # =====叠加CAM和原图，并保存图片=====
    # 1)读取原图
    src_image = cv2.imread(img_path)
    h, w, _ = src_image.shape
    # 2)cam转换成与原图大小一致的彩色度(cv2.COLORMAP_HSV为彩色图的其中一种类型)
    cam_color_resnet18 = cv2.applyColorMap(cv2.resize(cam_gray_resnet18, (w, h)),
                                        cv2.COLORMAP_HSV)
    cam_color_resnet50 = cv2.applyColorMap(cv2.resize(cam_gray_resnet50, (w, h)),
                                        cv2.COLORMAP_HSV)
    cam_color_densenet121 = cv2.applyColorMap(cv2.resize(cam_gray_densenet121, (w, h)),
                                            cv2.COLORMAP_HSV)
    # 3)合并cam和原图，并保存
    cam_resnet18 = src_image * 0.5 + cam_color_resnet18 * 0.5
    cam_resnet50 = src_image * 0.5 + cam_color_resnet50 * 0.5
    cam_densenet121 = src_image * 0.5 + cam_color_densenet121 * 0.5
    cam_hstack = np.hstack((src_image, cam_resnet18, cam_resnet50, cam_densenet121))
    cv2.imwrite(os.path.join(output_path,"result_"+img), cam_hstack)
    # 可视化
    Image.open(os.path.join(output_path,"result_"+img)).show()