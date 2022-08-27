# README

## 总览

CAM

- [x] CAM
- [x] Ablation-CAM
- [x] Eigen-CAM
- [x] EigenGrad-CAM
- [x] Full-Grad
- [x] Grad-CAM
- [x] Grad-CAM++
- [x] Grad-CAM-ElementWise
- [x] HiResCAM
- [x] Layer-CAM
- [x] RandomCAM
- [x] Score-CAM
- [x] XGrad-CAM
- [ ] Integrated Grad-CAM
- [x] Relevance-CAM

积分梯度

- [x] IG
- [x] IGOS
- [ ] Enhanced Integrated Gradients
- [ ] Guided IG

LIME

- [x] lime

LRP

- [x] LRP

## 样本阐述

**解释模型**：ResNet50、VGG19（权重及模型从Pytorch包直接导入）

**原始图片**：

![1](Images/1.JPEG)

![2](Images/2.JPEG)

![3](Images/3.JPEG)

![4](Images/4.JPEG)

**预测结果**：alp（高山，0.67）、jeep（吉普车，0.95）、jeep（吉普车，0.40）、dalmatian（达尔马提亚狗，0.999）

## 方法实况

> 左侧为VGG19，右侧为ResNet50

### CAM

![1](Show\CAM_1.JPEG)
![2](Show\CAM_2.JPEG)
![3](Show\CAM_3.JPEG)
![4](Show\CAM_4.JPEG)

### Ablation-CAM

![1](Results\Ablation-CAM\Ablation-CAM_cam_1.JPEG)
![2](Results\Ablation-CAM\Ablation-CAM_cam_2.JPEG)
![3](Results\Ablation-CAM\Ablation-CAM_cam_3.JPEG)
![4](Results\Ablation-CAM\Ablation-CAM_cam_4.JPEG)

### Eigen-CAM



### EigenGrad-CAM



### Full-Grad



### Grad-CAM

![1](Results\Grad-CAM\Grad-CAM_cam_1.JPEG)
![2](Results\Grad-CAM\Grad-CAM_cam_2.JPEG)
![3](Results\Grad-CAM\Grad-CAM_cam_3.JPEG)
![4](Results\Grad-CAM\Grad-CAM_cam_4.JPEG)

### Grad-CAM-ElementWise

![1](Results\Grad-CAM-ElementWise\Grad-CAM-ElementWise_cam_1.JPEG)
![2](Results\Grad-CAM-ElementWise\Grad-CAM-ElementWise_cam_2.JPEG)
![3](Results\Grad-CAM-ElementWise\Grad-CAM-ElementWise_cam_3.JPEG)
![4](Results\Grad-CAM-ElementWise\Grad-CAM-ElementWise_cam_4.JPEG)

### Grad-CAM++

![1](Results\Grad-CAM++\Grad-CAM++_cam_1.JPEG)
![2](Results\Grad-CAM++\Grad-CAM++_cam_2.JPEG)
![3](Results\Grad-CAM++\Grad-CAM++_cam_3.JPEG)
![4](Results\Grad-CAM++\Grad-CAM++_cam_4.JPEG)

### HiResCAM

![1](Results\HiResCAM\HiResCAM_cam_1.JPEG)
![2](Results\HiResCAM\HiResCAM_cam_2.JPEG)
![3](Results\HiResCAM\HiResCAM_cam_3.JPEG)
![4](Results\HiResCAM\HiResCAM_cam_4.JPEG)

### Layer-CAM



### RandomCAM

![1](Results\RandomCAM\RandomCAM_cam_1.JPEG)
![2](Results\RandomCAM\RandomCAM_cam_2.JPEG)
![3](Results\RandomCAM\RandomCAM_cam_3.JPEG)
![4](Results\RandomCAM\RandomCAM_cam_4.JPEG)

### Score-CAM

### XGrad-CAM

![1](Results\XGrad-CAM\XGrad-CAM_cam_1.JPEG)
![2](Results\XGrad-CAM\XGrad-CAM_cam_2.JPEG)
![3](Results\XGrad-CAM\XGrad-CAM_cam_3.JPEG)
![4](Results\XGrad-CAM\XGrad-CAM_cam_4.JPEG)

### Relevance-CAM

<center class="half">
    <img src="Results\Relevance-CAM\result_vgg19_1.JPEG" width="300"/>
    <img src="Results\Relevance-CAM\result_resnet50_1.JPEG" width="300"/>
</center>


<center class="half">
    <img src="Results\Relevance-CAM\result_vgg19_2.JPEG" width="300"/>
    <img src="Results\Relevance-CAM\result_resnet50_2.JPEG" width="300"/>
</center>


<center class="half">
    <img src="Results\Relevance-CAM\result_vgg19_3.JPEG" width="300"/>
    <img src="Results\Relevance-CAM\result_resnet50_3.JPEG" width="300"/>
</center>


<center class="half">
    <img src="Results\Relevance-CAM\result_vgg19_4.JPEG" width="300"/>
    <img src="Results\Relevance-CAM\result_resnet50_4.JPEG" width="300"/>
</center>

### LIME

<center class="half">
    <img src="Results\LIME\result_vgg19_1.JPEG" width="300"/>
    <img src="Results\LIME\result_resnet_1.JPEG" width="300"/>
</center>


<center class="half">
    <img src="Results\LIME\result_vgg19_2.JPEG" width="300"/>
    <img src="Results\LIME\result_resnet_2.JPEG" width="300"/>
</center>


<center class="half">
    <img src="Results\LIME\result_vgg19_3.JPEG" width="300"/>
    <img src="Results\LIME\result_resnet_3.JPEG" width="300"/>
</center>


<center class="half">
    <img src="Results\LIME\result_vgg19_4.JPEG" width="300"/>
    <img src="Results\LIME\result_resnet_4.JPEG" width="300"/>
</center>
### LRP

<center class="half">
    <img src="Results\LRP\result_desenet_1.JPEG" width="300"/>
    <img src="Results\LRP\result_resnet_1.JPEG" width="300"/>
</center>


<center class="half">
    <img src="Results\LRP\result_desenet_2.JPEG" width="300"/>
    <img src="Results\LRP\result_resnet_2.JPEG" width="300"/>
</center>


<center class="half">
    <img src="Results\LRP\result_desenet_3.JPEG" width="300"/>
    <img src="Results\LRP\result_resnet_3.JPEG" width="300"/>
</center>


<center class="half">
    <img src="Results\LRP\result_desenet_4.JPEG" width="300"/>
    <img src="Results\LRP\result_resnet_4.JPEG" width="300"/>
</center>