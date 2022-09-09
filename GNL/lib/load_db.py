# -*- coding: utf-8 -*-
""" Class definitions for using imagenet validation split."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import logging

import torch
import torchvision
from PIL import Image


class ImageNet:
  """ Class for ImageNet2012 database

  Attributes:
    __val_dir (str): Directory of validation split images
    __val_gt_path (str): Ground truth text file for val split
    __val_gt_txt_path (str): Text description of GT of val split
    __val_img_name_template (str): Naming convention for val split images
    __device (str): Device for torch
    val_label_list (list): List of GT for val split
    val_label_txt_list (list): List of txt description of GT for val split

  Properties:
    val_dir(str): Directory where validation split of ImageNet images
    val_gt_path(str): Ground truth text file for val split
    val_gt_txt_path(str): Text description of GT of val split
    val_img_name_template(str): Naming convention for val split images
    device(str): Device for torch
  """

  def __init__(self, 
    val_dir='', 
    val_gt_path = './GNL/res/ILSVRC2012_validation_ground_truth.txt',
    val_gt_txt_path = './GNL/res/ILSVRC2012_validation_ground_truth_text.txt',
    val_img_name_template = '{:d}.JPEG', 
    device='cuda'):

    # Set constants
    self.__device = device
    self.__val_dir = val_dir
    self.__val_gt_path = val_gt_path
    self.__val_gt_txt_path = val_gt_txt_path
    self.__val_img_name_template = val_img_name_template

    # Get label of imagenet2012 validation set
    self.val_label_list = self.__get_val_label_list()
    
    # Get Ground truth and corresponding text text
    self.val_label_txt_list = self.__get_val_label_txt_list()
    logging.info('ImageNet directory = {:s}'.format(self.val_dir))
  
  def __get_val_label_list(self):
    """ Get label of validation set in increasing order."""
    val_label_list = np.loadtxt(self.__val_gt_path)
    return val_label_list.astype(int)

  def __get_val_label_txt_list(self):
    """ Get label txt of validation set in increasing order."""
    with open(self.__val_gt_txt_path, 'r') as val_label_txt_list:
      val_label_txt_list = val_label_txt_list.readlines()
    for label_idx, label_txt in enumerate(val_label_txt_list):
      val_label_txt_list[label_idx] = label_txt.split(',')[0]

    return val_label_txt_list

  def pre_processing(self, image, with_normalization=True):
    """ Pre-processing PIL images for torchvision models

    Args:
      image (PIL Image): image will be processed
      with_normalization (bool): apply normalization of not
    """
    
    if with_normalization:
      normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225])
      transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        normalize,
        ])
    else:
      transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        ])
    return transforms(image).unsqueeze(dim=0).to(self.__device)

  def validation_data(self, data_idx):
    """ Get validation image w.r.t number

    Args:
      data_idx (int): number in 'ILSVRC2012_val_########.JPEG'
    """
    image_name = self.val_img_name_template.format(data_idx)
    image_path = os.path.join(self.__val_dir, image_name)
    image = Image.open(image_path).convert('RGB')

    label = self.val_label_list[data_idx - 1]
    label = torch.tensor(label).to(self.__device)
    label = label.unsqueeze(0).unsqueeze(0)
    label_txt = self.get_label_txt(label)

    return image, label, label_txt
    
  def get_label_txt(self, label_idx):
    """ Get text description of label with label_idx 

    Args:
      label_idx (int): index of label
    """    
    return self.val_label_txt_list[label_idx]


  @property
  def val_dir(self):
    return self.__val_dir

  @property
  def val_gt_path(self):
    return self.__val_gt_path

  @property
  def val_gt_txt_path(self):
    return self.__val_gt_txt_path

  @property
  def val_img_name_template(self):
    return self.__val_img_name_template

  @property
  def device(self):
    return self.__device

  @device.setter
  def device(self, device):
    self.__device = device