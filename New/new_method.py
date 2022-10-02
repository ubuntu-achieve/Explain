import os
import cv2
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torchvision import models
import  torch.nn.functional as F
from torch.autograd import Variable

import warnings
warnings.filterwarnings("ignore")

def feature_hook(model, input, output):
    # 用于记录梯度信息
    if use_cuda:
        feature_data.append(output.data.cpu().numpy())
    else:
        feature_data.append(output.data.numpy())

class GradCAM():
    
    def __init__(self, model_name='resnet50', use_cuda=False):
        self.feature_data = []
        self.use_cuda = use_cuda
        # 加载模型&&加载预训练模型
        if model_name == "resnet50":
            self.model = models.resnet50(pretrained=True)
        elif model_name == "vgg19":
            self.model = models.vgg19(pretrained=True)
        # 是否传入显存
        if use_cuda:
            self.model.cuda()
        # 开启预测
        self.model.eval()
        # 固定网络的底层，只反向传播误差不再计算导数
        for p in self.model.parameters():
            p.requires_grad = False
    
    def get_cam(self, img):
        results = self.model(img)
        results = F.softmax(results, dim=1)
        # 注册hook，捕获反向转播过程中流经该模块的梯度信息
        if model_name == 'resnet50':
            hand = self.model._modules.get('layer4').register_forward_hook(feature_hook)
            # 获取指定层的权重
            if self.use_cuda:
                fc_weights = self.model._modules.get('fc').weight.data.cpu().numpy()
                index      = np.argmax(results.data.cpu().numpy())
            else:
                fc_weights = self.model._modules.get('fc').weight.data.numpy()
                index      = np.argmax(results.data.numpy())
        elif model_name == 'vgg19':
            hand = self.model._modules.get('35').register_forward_hook(feature_hook)
            # 获取指定层的权重
            if self.use_cuda:
                fc_weights = self.model.classifier._modules['6'].weight.data.cpu().numpy()
                index      = np.argmax(results.data.cpu().numpy())
            else:
                fc_weights = self.model.classifier._modules['6'].weight.data.numpy()
                index      = np.argmax(results.data.numpy())
        # 计算CAM
        results[0][index].backward()
        feature = feature_data[-1]
        bz, nc, h, w = feature.shape
        # (512,) @ (512, 7*7) = (49,)
        cam = fc_weights[index].dot(feature.reshape(nc, h * w))
        cam = cam.reshape(h, w)  # (7, 7)
        # 归一化到[0, 1]之间
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        # 释放hook
        hand.remove()
        return cam

def load_model(model_name, use_cuda=False):
    '''
    加载模型
    
    model_name: 将要加载的模型(resnet50, vgg19)
    use_cuda: 是否使用显卡
    '''
    # 加载模型&&加载预训练模型
    if model_name == "resnet50":
        model = models.resnet50(pretrained=True)
    elif model_name == "vgg19":
        model = models.vgg19(pretrained=True)
    # 是否传入显存
    if use_cuda:
        model.cuda()
    # 开启预测
    model.eval()
    # 固定网络的底层，只反向传播误差不再计算导数
    for p in model.parameters():
        p.requires_grad = False
    return model

def tv_norm(input, tv_beta):
    # todo 
    img = input[0, 0, :]
    row_grad = torch.mean(torch.abs((img[:-1, :] - img[1:, :])).pow(tv_beta))
    col_grad = torch.mean(torch.abs((img[:, :-1] - img[:, 1:])).pow(tv_beta))
    return row_grad + col_grad

def topmaxPixel(HattMap, thre_num):
    # 获得倒数thre_num以前的值的索引(坐标形式)
    ii = np.unravel_index(np.argsort(HattMap.ravel())[: thre_num], HattMap.shape)
    #print(ii)
    OutHattMap = HattMap*0
    OutHattMap[ii] = 1

    img_ratio = np.sum(OutHattMap) / OutHattMap.size
    #print(OutHattMap.size)
    OutHattMap = 1 - OutHattMap
    return OutHattMap, img_ratio

def torch_to_numpy(tensor):
    '''
    将变量(Variable)转化为numpy的array
    '''
    if use_cuda:
        return np.transpose(tensor.data.cpu().numpy()[0], (1,2,0))
    else:
        return np.transpose(tensor.data.numpy()[0], (1,2,0))

def numpy_to_torch(img, use_cuda=False, requires_grad=False):
    '''
    将输入的图像拆分为单通道并转化封装为可反向传播的变量，或为掩码矩阵添加一维

    img: 输入的图片
    use_cuda: 是否使用GPU
    requires_grad: 是否记录梯度
    '''
    if len(img.shape) < 3:
        output = np.float32([img])
    else:
        output = np.transpose(img, (2, 0, 1))
    # 转为张量
    output = torch.from_numpy(output)
    # 是否存入GPU
    if use_cuda:
        output = output.cuda()
    # 最外层加一维
    output.unsqueeze_(0)
    # 封装tensor使其可以反向传播
    return Variable(output, requires_grad=requires_grad)

def preprocess_image(img, use_cuda=False, requires_grad=False):
    '''
    图像预处理，将其变为变量

    img:待处理的图像
    use_cuda:是否使用GPU
    requires_grad:是否记录梯度
    '''
    # 使用ImageNet的标准差和均值预处理图像
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    # 调换BGR为RGB
    pre_img = img.copy()[:,:,::-1]
    for i in range(3):
        pre_img[:,:,i] = (pre_img[:,:,i]-means[i])/stds[i]
    # 改变数组结构，拆分图片的三通道，并令行元素连续存储以加速计算
    pre_img = np.ascontiguousarray(np.transpose(pre_img, (2,0,1)))
    # 是否存入GPU
    if use_cuda:
        pre_img_tensor = torch.from_numpy(pre_img).cuda()
    else:
        pre_img_tensor = torch.from_numpy(pre_img)
    # 在最外层添加一维
    pre_img_tensor.unsqueeze_(0)
    # 封装tensor使其可以反向传播
    return Variable(pre_img_tensor, requires_grad=requires_grad)    

def Get_blurred_img(img_path, model, img_label=-1, resize_shape=(224, 224), param = ([51, 50], 11), blur_type= 'Gaussian', use_cuda = False):
    '''
    产生模糊图像作为基线
    img_path: 原始图像的路径
    model: 要可视化的模型
    img_label: 要可视化的分类目标(若值为-1则可视化概率最大的目标)
    resize_shape: the input size for the given model
    param: 高斯模糊、中值滤波的参数，格式为([高斯1,高斯2],中值)
    blur_type: 模糊的类型(Gaussian, Median, Mixed)
    use_cuda: 是否使用GPU
    '''

    img = cv2.imread(img_path, 1)
    img = cv2.resize(img, resize_shape)
    img = np.float32(img) / 255
    Kernelsize, SigmaX, Kernelsize_M = param[0][0], param[0][1], param[1]

    if blur_type =='Gaussian':   # 高斯模糊
        blurred_img = cv2.GaussianBlur(img, (Kernelsize, Kernelsize), SigmaX)

    elif blur_type == 'Median': # 中值滤波
        blurred_img = np.float32(cv2.medianBlur(img, Kernelsize_M)) / 255

    elif blur_type == 'Mixed': # 高斯模糊+中值滤波
        blurred_img1 = cv2.GaussianBlur(img, (Kernelsize, Kernelsize), SigmaX)
        blurred_img2 = np.float32(cv2.medianBlur(img, Kernelsize_M)) / 255
        blurred_img  = (blurred_img1 + blurred_img2) / 2
    # 图像预处理
    img_torch = preprocess_image(img, use_cuda, requires_grad = False)
    blurred_img_torch = preprocess_image(blurred_img, use_cuda, requires_grad = False)
    # 计算原始图像和模糊后图像的输出
    output         = model(img_torch)
    blurred_output = model(blurred_img_torch)

    if use_cuda:
        logitori = output.data.cpu().numpy().copy().squeeze()
        logitblur = blurred_output.data.cpu().numpy().copy().squeeze()
    else:
        logitori = output.data.numpy().copy().squeeze()
        logitblur = blurred_output.data.cpu().numpy().copy().squeeze()

    # 读取模型标签
    with open('imagenet_class_index.json', 'r') as f:
        model_label = json.load(f)
    top_5_idx    = np.argsort(logitori)[-5:][::-1]
    top_5_label  = [model_label[str(i)][1] for i in top_5_idx]
    top_5_values = [F.softmax(torch.tensor(logitori))[i] for i in top_5_idx]
    print('top_5_idx:', top_5_idx,'\ntop_5_label:', top_5_label, '\ntop_5_values:', top_5_values)
    # 寻找预测概率最大的标签
    output_label = top_5_idx[0]
    output_label_blur = np.where(logitblur == np.max(logitblur))[0][0]
    # 如果img_label=-1将选择概率最大的数作为可视化的标签
    if img_label == -1:
        img_label = output_label
    return img, blurred_img, logitori

def Integrated_Mask(img, blurred_img, model, model_name, category, max_iterations=15, integ_iter=20, tv_beta=2, l1_coeff=0.01*300, tv_coeff=0.2*300, size_init= 112, use_cuda=False):
    '''
    IGOS--使用集成梯度下降寻找最小、最光滑的区域，以最大程度的降低深度模型的输出

    img: 原始输入图像
    blurred_img: 输入图像基线(模糊后的图像)
    model: 要可视化的模型
    model_name: 要可视化的模型名称
    category: 要可视化的分类目标
    max_iterations: 最大迭代次数
    integ_iter: 用于计算集成梯度的最大点数
    tv_beta: 选择用于表示变化项的范数
    l1_coeff: L1范数的参数
    tv_coeff: TV范数的参数
    size_init: 生成掩码的分辨率
    use_cuda: 是否使用GPU
    '''
    cam = GradCAM(model_name,use_cuda)
    # 对输入图像和基线图像进行初始化
    img = preprocess_image(img, use_cuda, requires_grad=False)
    blurred_img = preprocess_image(blurred_img, use_cuda, requires_grad=False)
    # 记录图像尺寸
    resize_size = img.data.shape
    resize_wh = (img.data.shape[2], img.data.shape[3])
    # 是否转存到GPU
    if use_cuda:
        zero_img = Variable(torch.zeros(resize_size).cuda(), requires_grad=False)
    else:
        zero_img = Variable(torch.zeros(resize_size), requires_grad=False)
    # 初始化掩码矩阵
    mask_init = np.ones((size_init, size_init), dtype=np.float32)
    mask = numpy_to_torch(mask_init, use_cuda, requires_grad=True)
    # 初始化双线性差值计算对象
    if use_cuda:
        upsample = torch.nn.UpsamplingBilinear2d(size=resize_wh).cuda()
    else:
        upsample = torch.nn.UpsamplingBilinear2d(size=resize_wh)
    # 选择优化器，因为只需要求梯度所以可以使用各种优化器
    # idea 考虑全局设置学习率
    optimizer = torch.optim.Adam([mask], lr=0.1)
    #optimizer = torch.optim.SGD([mask], lr=0.1)
    # 进行预测并使用Softmax转换为概率
    target = torch.nn.Softmax(dim=1)(model(img))
    # 取出概率最大的类别索引
    if use_cuda:
        category_out = np.argmax(target.cpu().data.numpy())
    else:
        category_out = np.argmax(target.data.numpy())
    # 如果category=-1就选择概率最大的类别做可视化
    if category ==-1:
        category = category_out
    print("概率最高的类别", category_out)
    print("将要生成掩码的类别", category)
    print("Optimizing.. ")

    # 综合梯度下降
    curve2 = np.array([])
    alpha = 0.0001
    beta = 0.2
    # 开始迭代
    for i in tqdm(range(max_iterations)):
        # 将随机生成的掩码使用双线性插值扩充到图像大小
        upsampled_mask = upsample(mask)
        # 掩码要与RGB图像一同使用，将其扩充为三通道
        upsampled_mask = upsampled_mask.expand(
            1,
            3,
            upsampled_mask.size(2),
            upsampled_mask.size(3)
        )
        # L1范数和TV范数
        #loss1 = l1_coeff * torch.mean(torch.abs(1 - mask)) + tv_coeff * tv_norm(mask, tv_beta)
        # idea sum(M)
        #l0
        # loss1 = torch.tensor((1-mask).nonzero().shape[0]).cuda()
        #l1
        # loss1 = l1_coeff * torch.mean(torch.abs(1-mask))
        # l2
        loss1 = 0.003*torch.sum((1-mask).pow(2)).pow(0.5)/mask.shape[0]
        loss_all = loss1.clone()
        # 计算混合图像，令图像、扰动图像与掩码对应位置相乘得到混合图像
        # I⊙M+I'⊙(1-M)
        perturbated_input_base = img.mul(upsampled_mask) + blurred_img.mul(1 - upsampled_mask)
        # 注册hook，捕获反向转播过程中流经该模块的梯度信息
        if model_name == 'resnet50':
            hand = model._modules.get('layer4').register_forward_hook(feature_hook)
        elif model_name == 'vgg19':
            hand = model._modules.get('features').register_forward_hook(feature_hook)
        # 多点计算集成梯度
        for inte_i in range(integ_iter):
            # 扰动掩码，整体改变掩码矩阵的值
            integ_mask = 0.0 + ((inte_i + 1.0)/integ_iter) * upsampled_mask
            # 计算混合图像
            perturbated_input_integ = img.mul(integ_mask) + blurred_img.mul(1 - integ_mask)
            # 随机生成噪声
            noise = np.zeros((resize_wh[0], resize_wh[1], 3), dtype=np.float32)
            noise = noise + cv2.randn(noise, 0, 0.2)
            noise = numpy_to_torch(noise, use_cuda, requires_grad=False)
            # 加入噪声
            perturbated_input = perturbated_input_integ + noise
            new_image = perturbated_input
            # 重新预测并加上指定类别的输出
            outputs = torch.nn.Softmax(dim=1)(model(new_image))
            loss2 = outputs[0, category]
            # todo反向传播
            new_image = torch_to_numpy(new_image)
            new_image = preprocess_image(new_image, use_cuda=use_cuda, requires_grad=True)
            if use_cuda:
                cam_sum = torch.tensor(np.sum(cam.get_cam(new_image))).cuda()
            else:
                cam_sum = torch.tensor(np.sum(cam.get_cam(new_image)))
            loss_all = loss_all + loss2/integ_iter + cam_sum/integ_iter
            #print(loss2)
        # 计算给定类别的梯度、L1范数和TV范数的梯度
        optimizer.zero_grad()
        loss_all.backward()# main 反向传播
        # 更新参数
        optimizer.step()
        whole_grad = mask.grad.data.clone()
        #hand.remove()
        # 预测模糊图像
        loss2_ori = torch.nn.Softmax(dim=1)(model(perturbated_input_base))[0, category]
        # 模糊图像loss
        loss_ori = loss1 + loss2_ori
        # 记录loss
        if i==0:
            if use_cuda:
                curve2 = np.append(curve2, loss2_ori.data.cpu().numpy())
            else:
                curve2 = np.append(curve2, loss2_ori.data.numpy())
        # 模糊图像loss
        if use_cuda:
            loss_oridata = loss_ori.data.cpu().numpy()
        else:
            loss_oridata = loss_ori.data.numpy()

        # 修正Armijo线搜索准则
        step = 200.0
        MaskClone = mask.data.clone()
        MaskClone -= step * whole_grad
        MaskClone = Variable(MaskClone, requires_grad=False)
        MaskClone.data.clamp_(0, 1) # 令掩码的值在[0,1]
        # 这里的方向是whole_grad
        mask_LS = upsample(MaskClone)
        Img_LS = img.mul(mask_LS) + blurred_img.mul(1 - mask_LS)
        # 再次预测
        outputsLS = torch.nn.Softmax(dim=1)(model(Img_LS))
        # L1范数和TV范数
        #loss_LS = l1_coeff * torch.mean(torch.abs(1 - MaskClone)) + tv_coeff * tv_norm(MaskClone, tv_beta) + outputsLS[0, category]
        # idea sum(M)
        Img_LS  = torch_to_numpy(Img_LS)
        Img_LS  = preprocess_image(Img_LS, use_cuda=use_cuda, requires_grad=True)
        if use_cuda:
            cam_sum = torch.tensor(np.sum(cam.get_cam(Img_LS))).cuda()
        else:
            cam_sum = torch.tensor(np.sum(cam.get_cam(Img_LS)))
        # l0
        #loss_LS = (1-MaskClone).nonzero().shape[0] + outputsLS[0, category] + cam_sum
        # l1
        #loss_LS = torch.mean(torch.abs(1-MaskClone)) + outputsLS[0, category] + cam_sum
        # l2
        loss_LS = 0.003*torch.sum((1-MaskClone).pow(2)).pow(0.5)/MaskClone.shape[0] + outputsLS[0, category] + cam_sum

        if use_cuda:
            loss_LSdata = loss_LS.data.cpu().numpy()
        else:
            loss_LSdata = loss_LS.data.numpy()

        new_condition = whole_grad ** 2
        new_condition = new_condition.sum()
        new_condition = alpha * step * new_condition

        while loss_LSdata > loss_oridata - new_condition.cpu().numpy():
            step *= beta

            MaskClone = mask.data.clone()
            MaskClone -= step * whole_grad
            MaskClone = Variable(MaskClone, requires_grad=False)
            MaskClone.data.clamp_(0, 1)
            mask_LS = upsample(MaskClone)
            Img_LS = img.mul(mask_LS) + blurred_img.mul(1 - mask_LS)
            outputsLS = torch.nn.Softmax(dim=1)(model(Img_LS))
            #loss_LS = l1_coeff * torch.mean(torch.abs(1 - MaskClone)) + tv_coeff * tv_norm(MaskClone, tv_beta) + outputsLS[0, category]
            # idea
            Img_LS = torch_to_numpy(Img_LS)
            Img_LS = preprocess_image(Img_LS, use_cuda=use_cuda, requires_grad=True)
            if use_cuda:
                cam_sum = torch.tensor(np.sum(cam.get_cam(Img_LS))).cuda()
            else:
                cam_sum = torch.tensor(np.sum(cam.get_cam(Img_LS)))
            # l0
            #loss_LS = (1-MaskClone).nonzero().shape[0] + outputsLS[0, category] + cam_sum
            # l1
            #loss_LS = torch.mean(torch.abs(1-MaskClone)) + outputsLS[0, category] + cam_sum
            # l2
            loss_LS = 0.003*torch.sum((1-MaskClone).pow(2)).pow(0.5)/MaskClone.shape[0] + outputsLS[0, category] + cam_sum

            if use_cuda:
                loss_LSdata = loss_LS.data.cpu().numpy()
            else:
                loss_LSdata = loss_LS.data.numpy()

            new_condition = whole_grad ** 2  # Here the direction is the whole_grad
            new_condition = new_condition.sum()
            new_condition = alpha * step * new_condition

            if step<0.00001:
                break

        mask.data -= step * whole_grad
        #######################################################################################################

        if use_cuda:
            curve2 = np.append(curve2, loss2_ori.data.cpu().numpy())
        else:
            curve2 = np.append(curve2, loss2_ori.data.numpy())

        mask.data.clamp_(0, 1)

        if max_iterations >3:

            if i == int(max_iterations / 2):
                if np.abs(curve2[0] - curve2[i]) <= 0.001:
                    #print('Adjust Parameter l1_coeff at iteration:', int(max_iterations / 2))
                    l1_coeff = l1_coeff / 10

            elif i == int(max_iterations / 1.25):
                if np.abs(curve2[0] - curve2[i]) <= 0.01:
                    #print('Adjust Parameters l1_coeff again at iteration:', int(max_iterations / 1.25))
                    l1_coeff = l1_coeff / 5

    upsampled_mask = upsample(mask)

    if use_cuda:
        mask = mask.data.cpu().numpy().copy()
    else:
        mask = mask.data.numpy().copy()

    return mask, upsampled_mask, curve2, category

use_cuda     = True
img_label    = -1
input_path   = './Images'
output_path  = './Results/New'
model_name   = 'resnet50'
feature_data = []  # 存储梯度信息
if __name__ == "__main__":
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    model = load_model(model_name, use_cuda=use_cuda)
    for img_path in os.listdir(input_path)[3:4]:
        img, blurred_img, logitori = Get_blurred_img(
            os.path.join(input_path, img_path),
            model,
            img_label,
            resize_shape=(224, 224),
            param=([51, 50],11),
            blur_type='Gaussian',
            use_cuda=use_cuda
        )
        mask, upsampled_mask, curve2, category = Integrated_Mask(
            img,
            blurred_img,
            model,
            model_name,
            img_label,
            max_iterations=30,
            integ_iter=20,
            tv_beta=2,
            l1_coeff=0.01 * 100,
            tv_coeff=0.2 * 100,
            size_init=28,
            use_cuda=use_cuda
        )
        # 生成热力图
        img = cv2.imread(os.path.join(input_path, img_path))
        if use_cuda:
            cam = upsampled_mask[0,0].data.cpu().numpy()
        else:
            cam = upsampled_mask[0,0].data.numpy()
        cam = (cam.max() - cam) / (cam.max() - cam.min())
        cam_gray = np.uint8(255 * cam)
        cam_gray = cv2.resize(cam_gray, img.shape[:-1][::-1])
        #plt.imshow(cam_gray)
        cam_RGB = np.transpose(np.array([cam_gray, cam_gray, cam_gray]), (1,2,0))
        cam_RGB = img*0.5+0.5*cv2.applyColorMap(cam_RGB, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(output_path,"result_"+img_path), cam_RGB)
