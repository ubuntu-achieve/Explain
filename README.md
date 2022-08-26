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
- [ ] Relevance-CAM

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

**解释模型**：ResNet50（权重及模型从Pytorch包直接导入）

**原始图片**：

![1](Images/1.JPEG)

![2](Images/2.JPEG)

![3](Images/3.JPEG)

![4](Images/4.JPEG)

**预测结果**：alp（高山，0.67）、jeep（吉普车，0.95）、jeep（吉普车，0.40）、dalmatian（达尔马提亚狗，0.999）