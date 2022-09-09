# -*- coding: utf-8 -*- 
""" Implementation of proposed Guided Non-linearity method.
Our implementation based on the PyTorch Captum library (https://captum.ai/)
We attached backward hook to ReLU gradients vaules of non-linear units.

We apply proposed method on Integrated Gradients, however,
our implementation could be easily extended to the other existing method that
based on the backpropagation of gradients. (e.g. Excitation backprop, Grad-CAM)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys
sys.path.append("./GNL/lib")

import numpy as np
import argparse
import logging
import torch
import torch.nn.functional as F
from datetime import datetime
from captum.attr import (
  IntegratedGradients,
)


class GuidedIntgratedGradientsHook:
  """ Add hooks for Guided Integrated Gradients (GIG).

  Attributes:
    forward_hooks (list): List of forward hooks.
    backward_hooks (list): List of backward hooks.
  """
  def __init__(self):
    self.backward_hooks = []

  def _register_hooks(self, module):
    """ Registrer hooks for GIG.

    Args:
      module (torch.nn.Module): Module for add hooks.
    """
    if isinstance(module, torch.nn.ReLU):
      self.backward_hooks.append(
        module.register_backward_hook(self._backward_hook_relu)
      )
    if isinstance(module, torch.nn.ReLU6):
      self.backward_hooks.append(
        module.register_backward_hook(self._backward_hook_relu)
      )        
    if isinstance(module, torch.nn.MaxPool2d):
      self.backward_hooks.append(
        module.register_backward_hook(self._backward_hook_maxpool)
      )

  def _backward_hook_relu(self, module, grad_input, grad_output):
    """ Backward hook of ReLU for GIG.
    https://pytorch.org/docs/stable/nn.html#torch.nn.Module.register_backward_hook

    Args:
      module (torch.nn.Module): Module for add hooks.
      grad_input (tuple of torch.tensor): Input for grad_ftn from previous
        layer.
      grad_output (tuple of torch.tensor): Output for grad_ftn from previous
        layer.
    """
    if isinstance(grad_output, tuple):
      guided_grad_input = tuple(F.relu(g_in) for g_in in grad_input)
    else:
      guided_grad_input = F.relu(grad_input)
    return guided_grad_input


  def _backward_hook_maxpool(self, module, grad_input, grad_output):
    """ Backward hook of Maxpool2d for GIG.
    https://pytorch.org/docs/stable/nn.html#torch.nn.Module.register_backward_hook

    Args:
      module (torch.nn.Module): Module for add hooks.
      grad_input (tuple of torch.tensor): Input for grad_ftn from previous
        layer.
      grad_output (tuple of torch.tensor): Output for grad_ftn from previous
        layer.
    """
    if isinstance(grad_input, tuple):
      guided_grad = tuple(
        F.interpolate((g_out > 0.0).float(), size=g_in.shape[2]) * g_in 
        for g_out, g_in in zip(grad_output, grad_input))
    return guided_grad


  def _remove_hooks(self):
    """ Remove all hooks. """
    for hook in self.backward_hooks:
      hook.remove()


class GetAttrubutions:
  """ Get attribution for given model (torchvision.models) w.r.t input image
  and given class index.

  Attributes:
    method (str): Name of attribution methods.
    model(torchvision.models object): Model to calculate attribution.
    __attr_obj (captum.attr): Captum object for calculating attribution.
    __call_ftn (function object): Captum attribution method.
      torchvision.models with modification on
      AdaptiveAvgPool -> AdaptiveMaxPool2d
  """
  def __init__(self, method, model):
    """ Init class.

    Args:
      method (str): Name of attribution methods.
      model(torchvision.models object): Model to calculate attribution.
    """
    self.method = method
    self.__attr_obj = None
    self.__call_ftn = None

    self.__attr_obj, self.__call_ftn = self.__get_attribution_objects(
      method=method,
      model=model,
    )
    if self.__attr_obj == None or self.__call_ftn == None:
      logging.error('Error in init.')


  def __call__(self, input_feature, target):
    """ Call method.
    Calculate attribution according to self.method on model.

    Args:
      input_feature (torch.tensor): Input for network.
      target (torch.tensor): Class index for calculating attribution.
    """
    return self.__call_ftn(input_feature, target)

  def __get_attribution_objects(self, method, model):
    """ Get catpum.attr.attribution method object.

    Args:
      method (str): Name of attribution methods.
      model(torchvision.models object): Model to calculate attribution.
    """
    if method == 'guided_integrated_gradients':
      attr_obj = self.__get_guided_integrated_gradients_obj(model)
      call_ftn = self.__guided_integrated_gradients
    elif method == 'integrated_gradients':
      attr_obj = self.__get_integrated_gradients_obj(model)
      call_ftn = self.__integrated_gradients

    return attr_obj, call_ftn

  def __get_guided_integrated_gradients_obj(self, model):
    """ Get captum.attr.IntegratedGradients object and apply
    Maxpool and ReLU gradient hook for Guided Integrated Gradients.

    Args:
      model(torchvision.models object): Model to calculate attribution.
    """
    gig_gradient_hook = GuidedIntgratedGradientsHook()
    model.apply(gig_gradient_hook._register_hooks)

    return IntegratedGradients(model)


  def __get_integrated_gradients_obj(self, model):
    """ Get captum.attr.IntegratedGradients object.

    Args:
      model(torchvision.models object): Model to calculate attribution.
    """
    return IntegratedGradients(model)


  def __guided_integrated_gradients(self, input_feature, target):
    """ Guided Integrated Gradients.
    Same as captum.attr.integrated_gradients except use models with
    backward gradient hook on ReLU layers.

    Args:
      input_feature (torch.tensor): Input for network.
      target (torch.tensor): Class index for calculating attribution.
    """
    attributions = self.__attr_obj.attribute(
      input_feature,
      target=target,
      return_convergence_delta=False,
      )
    return attributions


  def __integrated_gradients(self, input_feature, target):
    """ Integrated Gradients.

    Options for captum.attr.integrated_gradients are
      'baselines=torch.zeros_like(input)' and 'return_convergence_delta=False'.
    Where,
      baselines: In the cases when `baselines` is not provided, we internally
                 use zero scalar corresponding to each input tensor.
      return_convergence_delta (bool, optional): Indicates whether to return
                 convergence delta or not.
    https://github.com/pytorch/captum/blob/master/captum/attr/_core/integrated_gradients.py


    Args:
      input_feature (torch.tensor): Input for network.
      target (torch.tensor): Class index for calculating attribution.
    """
    attributions = self.__attr_obj.attribute(
      input_feature,
      target=target,
      return_convergence_delta=False,
      )
    return attributions