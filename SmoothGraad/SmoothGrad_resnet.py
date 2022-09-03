import os
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import models, transforms

import saliency.core as saliency
# 消除警告
import warnings
warnings.filterwarnings('ignore')

def ShowImage(im, title='', ax=None):
    if ax is None:
        plt.figure()
    plt.axis('off')
    plt.imshow(im)
    plt.title(title)

def LoadImage(file_path):
    im = PIL.Image.open(file_path)
    im = im.resize((299, 299))
    im = np.asarray(im)
    return im

transformer = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
def PreprocessImages(images):
    images = np.array(images)
    images = images/255
    images = np.transpose(images, (0,3,1,2))
    images = torch.tensor(images, dtype=torch.float32)
    images = transformer.forward(images)
    return images.requires_grad_(True)

def conv_layer_forward(m, i, o):
    # move the RGB dimension to the last dimension
    conv_layer_outputs[saliency.base.CONVOLUTION_LAYER_VALUES] = torch.movedim(o, 1, 1).detach().numpy()
def conv_layer_backward(m, i, o):
    # move the RGB dimension to the last dimension
    conv_layer_outputs[saliency.base.CONVOLUTION_OUTPUT_GRADIENTS] = torch.movedim(o[0], 1, 1).detach().numpy()

def call_model_function(images, call_model_args=None, expected_keys=None):
    images = PreprocessImages(images)
    target_class_idx =  call_model_args[class_idx_str]
    output = model(images)
    m = torch.nn.Softmax(dim=1)
    output = m(output)
    if saliency.base.INPUT_OUTPUT_GRADIENTS in expected_keys:
        outputs = output[:,target_class_idx]
        grads = torch.autograd.grad(outputs, images, grad_outputs=torch.ones_like(outputs))
        grads = torch.movedim(grads[0], 1, 3)
        gradients = grads.detach().numpy()
        return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}
    else:
        one_hot = torch.zeros_like(output)
        one_hot[:,target_class_idx] = 1
        model.zero_grad()
        output.backward(gradient=one_hot, retain_graph=True)
        return conv_layer_outputs

class_idx_str = 'class_idx_str'
root_path = 'Images'
output_path = "Results/SoomthGrad"

if __name__ == "__main__":
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    model = models.resnet50(pretrained=True)
    eval_mode = model.eval()

    conv_layer = model.fc
    conv_layer_outputs = {}
    conv_layer.register_forward_hook(conv_layer_forward)
    conv_layer.register_full_backward_hook(conv_layer_backward)
    for img in os.listdir(root_path):
        im_orig = LoadImage(os.path.join(root_path, img))
        im_tensor = PreprocessImages([im_orig])

        predictions = model(im_tensor)
        predictions = predictions.detach().numpy()
        prediction_class = np.argmax(predictions[0])
        call_model_args = {class_idx_str: prediction_class}

        print("Prediction class: " + str(prediction_class))  # Should be a doberman, class idx = 236
        im = im_orig.astype(np.float32)

        # Construct the saliency object. This alone doesn't do anthing.
        gradient_saliency = saliency.GradientSaliency()

        # Compute the vanilla mask and the smoothed mask.
        smoothgrad_mask_3d = gradient_saliency.GetSmoothedMask(im, call_model_function, call_model_args)

        # Call the visualization methods to convert the 3D tensors to 2D grayscale.
        smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(smoothgrad_mask_3d)
        plt.imsave(os.path.join(output_path, "SoomthGrad_resnet_%s"%(img)), smoothgrad_mask_grayscale, cmap=plt.cm.gray)
