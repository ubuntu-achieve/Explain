#coding=utf-8
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

def imshow2(hm,imgtensor,fns,q=100, is_show=True):

    def invert_normalize(ten, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
      print(ten.shape)
      s=torch.tensor(np.asarray(std,dtype=np.float32)).unsqueeze(1).unsqueeze(2)
      m=torch.tensor(np.asarray(mean,dtype=np.float32)).unsqueeze(1).unsqueeze(2)

      res=ten*s+m
      return res

    def showimgfromtensor(inpdata):

      ts=invert_normalize(inpdata)
      a=ts.data.squeeze(0).numpy()
      saveimg=(a*255.0).astype(np.uint8)

      #PIL.Image.fromarray(np.transpose(saveimg,[1,2,0]), 'RGB').show() #.save(savename)
    ######## 


    fig, axs = plt.subplots(1, 2 )

    hm = hm.squeeze().sum(dim=0).numpy()

    clim = np.percentile(np.abs(hm), q)
    hm = hm / clim
    #hm = gregoire_black_firered(hm)
    #axs[1].imshow(hm)
    axs[1].imshow(hm, cmap="seismic", clim=(-1, 1))
    axs[1].axis('off')

    ts=invert_normalize(imgtensor.squeeze())
    a=ts.data.numpy().transpose((1, 2, 0))
    axs[0].imshow(a)
    axs[0].axis('off')
    if is_show:
      plt.show()
    else:
      plt.imsave(fns, hm, cmap="seismic")
