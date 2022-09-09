# -*- coding: utf-8 -*-
""" Class definitions to ease loading torchvision models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import torch
import torchvision

class TorchVisionModels:
  """ Get torchvision.models.
  vgg16, vgg19, resnet34, resnet50, googlenet

  Attributes:
    __model_name (str): Name of torchvision model
    __device (str): Device for torch

  Properties:
    model_name (str): self.__model_name
    device (str): self.__device
  """

  def __init__(self, model_name, device='cuda'):
    """ Initialization

    Args:
      model_name (str): Name of torchvision model
      device (str): Device for torch
    """
    self.__model_name = model_name
    self.__device = device

    logging.info('Model Name = {:s}'.format(model_name))

  def pretrained_model(self):
    """ Get pretrained model from torchvision.models.

    Returns:
      __pretrained_model (torchvision.models): Pretrained model from
        torchvision.models.
    """
    # Load pre-trained models in torchvision.models
    if self.__model_name == 'VGG16' or self.__model_name == 'vgg16':
      pretrained_model = torchvision.models.vgg16(pretrained=True)
    if self.__model_name == 'VGG19' or self.__model_name == 'vgg19':
      pretrained_model = torchvision.models.vgg19(pretrained=True)
    if self.__model_name == 'ResNet34' or self.__model_name == 'resnet34':
      pretrained_model = torchvision.models.resnet34(pretrained=True)
    if self.__model_name == 'ResNet50' or self.__model_name == 'resnet50':
      pretrained_model = torchvision.models.resnet50(pretrained=True)
    if self.__model_name == 'GoogleNet' or self.__model_name == 'googlenet':
      pretrained_model = torchvision.models.googlenet(pretrained=True)
      from torchvision.models.googlenet import (
        BasicConv2d as googlenet_basic_conv_2d,
      )
      def replace_functional_relu_to_module_relu(module):
        if isinstance(module, googlenet_basic_conv_2d):
          setattr(module,
            'bn',
            torch.nn.Sequential(module.bn, torch.nn.ReLU())
          )
      pretrained_model.apply(replace_functional_relu_to_module_relu)

    # Let model in evaluation mode and proper device
    pretrained_model = pretrained_model.to(self.__device).eval()

    return pretrained_model


  @property
  def model_name(self):
    return self.__model_name

  @property
  def device(self):
    return self.__device

  @device.setter
  def device(self, device):
    self.__device = device