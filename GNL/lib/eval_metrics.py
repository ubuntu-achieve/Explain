# -*- coding: utf-8 -*-
""" Evaluation metrics (deletion and insertion scores) used in paper.
We used the authors' code[1] of RISE[2] that proposed deletion/insertion scores.
--------------------------------------------------------------------------------
[1] https://github.com/eclique/RISE
[2] Petsiuk, Vitali, Abir Das, and Kate Saenko. 
    "Rise: Randomized input sampling for explanation of black-box models." 
    arXiv preprint arXiv:1806.07421 (2018).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import scipy.io
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import torch

# Constants
HW = 224 * 224  # Image area
n_classes = 1000  # Number of classes in ImageNet


def gkern(klen, nsig):
  """Returns a Gaussian kernel array.
  Convolution with it results in image blurring."""
  # create nxn zeros
  inp = np.zeros((klen, klen))
  # set element at the middle to one, a dirac delta
  inp[klen//2, klen//2] = 1
  # gaussian-smooth the dirac, resulting in a gaussian filter mask
  k = gaussian_filter(inp, nsig)
  kern = np.zeros((3, 3, klen, klen))
  kern[0, 0] = k
  kern[1, 1] = k
  kern[2, 2] = k
  return torch.from_numpy(kern.astype('float32'))

def auc(arr):
  """Returns normalized Area Under Curve of the array."""
  return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)

class CausalMetric():

  def __init__(self, model, mode, step, substrate_fn):
    """Create deletion/insertion metric instance.

    Args:
      model (nn.Module): Black-box model being explained.
      mode (str): 'del' or 'ins'.
      step (int): number of pixels modified per one iteration.
      substrate_fn (func): a mapping from old pixels to new pixels.
    """
    assert mode in ['del', 'ins']
    self.model = model
    self.mode = mode
    self.step = step
    self.substrate_fn = substrate_fn

  def single_run(self, img_tensor, explanation):
    """Run metric on one image-saliency pair.

    Args:
      img_tensor (Tensor): normalized image tensor.
      explanation (np.ndarray): saliency map.

    Return:
      scores (nd.array): Array containing scores at every step.
    """
    pred = self.model(img_tensor.cuda())
    top, c = torch.max(pred, 1)
    c = c.cpu().numpy()[0]
    n_steps = (HW + self.step - 1) // self.step

    if self.mode == 'del':
      start = img_tensor.clone()
      finish = self.substrate_fn(img_tensor)
    elif self.mode == 'ins':
      start = self.substrate_fn(img_tensor)
      finish = img_tensor.clone()

    scores = np.empty(n_steps + 1)
    # Coordinates of pixels in order of decreasing saliency
    salient_order = np.flip(
      np.argsort(explanation.reshape(-1, HW).numpy(), axis=1), axis=-1)
    for i in range(n_steps+1):
      pred = self.model(start.cuda())
      pr, cl = torch.topk(pred, 2)
      scores[i] = pred[0, c]
      if i < n_steps:
        coords = salient_order[:, self.step * i:self.step * (i + 1)]
        start.cpu().numpy().reshape(1, 3, HW)[0, :, coords] = \
          finish.cpu().numpy().reshape(1, 3, HW)[0, :, coords]
    return scores

  def evaluate(self, img_batch, exp_batch, batch_size):
    """Efficiently evaluate big batch of images.

    Args:
      img_batch (Tensor): batch of images.
      exp_batch (np.ndarray): batch of explanations.
      batch_size (int): number of images for one small batch.

    Returns:
      scores (nd.array): Array containing scores at every step for every image.
    """
    n_samples = img_batch.shape[0]
    predictions = torch.FloatTensor(n_samples, n_classes)
    assert n_samples % batch_size == 0
    for i in range(n_samples // batch_size):
      preds = self.model(img_batch[i*batch_size:(i+1)*batch_size].cuda()).cpu()
      predictions[i*batch_size:(i+1)*batch_size] = preds
    top = np.argmax(predictions.detach(), -1)
    n_steps = (HW + self.step - 1) // self.step
    scores = np.empty((n_steps + 1, n_samples))
    salient_order = np.flip(np.argsort(
      exp_batch.reshape(-1, HW), axis=1).numpy(), axis=-1)
    r = np.arange(n_samples).reshape(n_samples, 1)

    substrate = torch.zeros_like(img_batch)
    for j in range(n_samples // batch_size):
      substrate[j*batch_size:(j+1)*batch_size] = \
        self.substrate_fn(img_batch[j*batch_size:(j+1)*batch_size])

    if self.mode == 'del':
      caption = 'Deleting  '
      start = img_batch.clone()
      finish = substrate
    elif self.mode == 'ins':
      caption = 'Inserting '
      start = substrate
      finish = img_batch.clone()

    # While not all pixels are changed
    for i in range(n_steps+1):
      # Iterate over batches
      for j in range(n_samples // batch_size):
        # Compute new scores
        preds = self.model(start[j*batch_size:(j+1)*batch_size].cuda()).detach()
        preds = preds.cpu().numpy()[
          range(batch_size), top[j*batch_size:(j+1)*batch_size]]
        scores[i, j*batch_size:(j+1)*batch_size] = preds
      # Change specified number of most salient pixels to substrate pixels
      coords = salient_order[:, self.step * i:self.step * (i + 1)]
      start.cpu().numpy().reshape(n_samples, 3, HW)[r, :, coords] = \
        finish.cpu().numpy().reshape(n_samples, 3, HW)[r, :, coords]

    return scores