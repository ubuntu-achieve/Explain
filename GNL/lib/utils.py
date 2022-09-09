# -*- coding: utf-8 -*-
""" Utility  functions """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import scipy.io
import cv2
import torch
from matplotlib import pyplot as plt

from lib.eval_metrics import auc


def torch_to_np(tensor):
  """ Return torch tensor as numpy array.

  Args:
    tensor (torch.tensor): Input tensor.
  """

  return tensor.clone().detach().cpu().numpy()

def save_as_mat(output_dir, mat_data, mat_name):
  """ Save mat_data as mat

  Args:
    output_dir (str): Output dir
    mat_data (np.array): Data to save as mat file
    mat_name (str): Name of output mat file
  """
  scipy.io.savemat(os.path.join(output_dir, '{}.mat'.format(mat_name)),
                   {mat_name.split('.')[0]: mat_data.squeeze()})


def save_as_heatmap(output_dir, heatmap, img_name):
  """ Save np.array as saliency image.

  Args:
    output_dir (string): Output dir.
    heatmap (np.array): The heatmap of importace for each pixel.
    img_name (string): Name of output file.
  """
  img_path = os.path.join(output_dir, img_name)
  heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

  cv2.imwrite(img_path, heatmap)


def draw_deletion_score_graph(del_score, output_dir, img_name):
  """ Draw deletion scores 
  
  Args:
    del_score (np.array): Array of prediction probability
    output_dir (string): Path of output directory
    img_name (string): Name of output file.
  """
  color = 'C1'
  fontsize = 18
  plt.figure(figsize=(5, 5))
  plt.plot(np.arange(225) / 224, del_score[:225], color=color)
  plt.xlim(-0.01, 1.01)
  plt.ylim(0, 1.05)
  plt.fill_between(np.arange(225) / 224, 0, del_score[:225], 
    alpha=0.4,
    color=color,
  )
  plt.xticks(fontsize=fontsize)
  plt.yticks(fontsize=fontsize)
  plt.text(.3, .95, 'AUC = {:.3f}'.format(auc(del_score)), fontsize=fontsize)
  plt.tight_layout()    
  plt.savefig(os.path.join(output_dir, img_name) + '.deletion.png')
  plt.close()


def draw_insertion_score_graph(ins_score, output_dir, img_name):
  """ Draw insertion scores
  Args:
    ins_score (np.array): Array of prediction probability
    output_dir (string): Path of output directory
    img_name (string): Name of output file.
  """
  color = 'C2'
  fontsize = 18
  plt.figure(figsize=(5, 5))    
  plt.plot(np.arange(225) / 224, ins_score[:225], color=color)
  plt.xlim(-0.01, 1.01)
  plt.ylim(0, 1.05)
  plt.fill_between(np.arange(225) / 224, 0, ins_score[:225], 
    alpha=0.4,
    color=color,
  )
  plt.xticks(fontsize=fontsize)
  plt.yticks(fontsize=fontsize)    
  plt.text(.3, .95, 'AUC = {:.3f}'.format(auc(ins_score)), fontsize=fontsize)
  plt.tight_layout() 
  plt.savefig(os.path.join(output_dir, img_name) + '.insertion.png')
  plt.close()