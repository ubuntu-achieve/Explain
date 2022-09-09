# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import sys
#sys.path.append("src/lib")

import numpy as np
import argparse
import matplotlib.pylab as plt

import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from captum.attr import (
  IntegratedGradients,
)

import lib.load_nn as load_nn
import lib.load_db as load_db
from lib.eval_metrics import (
  gkern,
  CausalMetric,
)
from lib.utils import (
  torch_to_np,
  save_as_heatmap,
  save_as_mat,
  draw_deletion_score_graph,
  draw_insertion_score_graph,
)
from gnl import GetAttrubutions

# Argument parsing.
parser = argparse.ArgumentParser(description='Set Exp Constants')
parser.add_argument('--gpu_num', '-gn', default=0, type=int)
parser.add_argument('--data_dir', '-da', default='data', type=str)
parser.add_argument('--start', '-si', default=1, type=int)
parser.add_argument('--end', '-ei', default=50000, type=int)
parser.add_argument('--stride', '-ie', default=10, type=int)
parser.add_argument('--model_name', '-mn', default='resnet50', type=str)
parser.add_argument('--method', '-md',
                    default='guided_integrated_gradients',
                    #default='integrated_gradients',
                    type=str)
exp_args = parser.parse_args()

# Env settings
os.environ['CUDA_VISIBLE_DEVICES'] = str(exp_args.gpu_num)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(exp_args, device):
  """ Calculate deletion/insertion score of attribution heatmap by 
  integrated gradients or guided non-linearity.

  Args:
    exp_args (argparse.args): arguments for exp
    device (str): device for exp
  """
  torch.autograd.set_detect_anomaly(True)

  # Define device, model and db.
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = load_nn.TorchVisionModels(
    model_name=exp_args.model_name,
    device=device,
  ).pretrained_model()
  imagenet = load_db.ImageNet(
    val_dir=exp_args.data_dir,
    device=device,
  )

  # Get deletion / insertion metric object
  del_metric = CausalMetric(
    model=torch.nn.Sequential(model, torch.nn.Softmax(dim=1)), 
    mode='del', 
    step=224, 
    substrate_fn=torch.zeros_like)
  ins_metric = CausalMetric(
    model=torch.nn.Sequential(model, torch.nn.Softmax(dim=1)),
    mode='ins', 
    step=224, 
    substrate_fn=lambda x: F.conv2d(x, gkern(11, 5), padding=5))

  # List of failed samples
  failed_sample_list = []

  # Obj to calculate attributions
  get_attribution = GetAttrubutions(
    method=exp_args.method,
    model=model,
  )

  # Inverse each prediction
  for i, data_idx in enumerate(
    range(exp_args.start, exp_args.end, exp_args.stride)):

    print('Processing {:08d}th file {}'.format(i, data_idx))

    # Forward pass
    input_img, label, label_txt = imagenet.validation_data(data_idx)
    processed_input = imagenet.pre_processing(input_img)
    logits = model(processed_input)
    argmax_logit = torch.argmax(logits).reshape(label.shape)

    # Get attrubution and heatmap.
    attr = get_attribution(
      input_feature=processed_input,
      target=argmax_logit.to(device),
    )
    heatmap = attr.sum(dim=1).abs().unsqueeze(0)
    heatmap = heatmap / heatmap.max()

    # Calculate deletion metric
    del_score = del_metric.single_run(
      img_tensor=processed_input.cpu(),
      explanation=heatmap.cpu())

    # Calculate insertion metric
    ins_score = ins_metric.single_run(
      img_tensor=processed_input.cpu(),
      explanation=heatmap.cpu())

    # Save results.
    result_name = 'result_%s'%(str(exp_args.model_name))+imagenet.val_img_name_template
    result_name = result_name.format(data_idx)
    print('save"%s"to%s'%(result_name, output_path))

    # Save raw input image
    # save_image(
    #   imagenet.pre_processing(input_img, with_normalization=False),
    #   os.path.join(exp_output_dir, result_name) + '.cropped_input.png'
    # )

    # Save heatmap as matlab mat file
    # save_as_mat(
    #   output_dir=exp_output_dir,
    #   mat_data=np.transpose(torch_to_np(heatmap).squeeze(0), (1, 2, 0)),
    #   mat_name=result_name,
    # )

    # Save heatmap as png
    save_as_heatmap(
      output_dir=output_path,
      heatmap=heatmap.cpu().numpy().squeeze(),
      img_name=result_name,
    )

    # Draw deletion/insertin score graph
    draw_deletion_score_graph(del_score, output_path, result_name)
    draw_insertion_score_graph(ins_score, output_path, result_name)

output_path = './Results/GNL'
if __name__ == '__main__':
  if not os.path.isdir(output_path):
    os.makedirs(output_path)
  main(exp_args=exp_args, device=device)