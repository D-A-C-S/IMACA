from kornia.filters import gaussian_blur2d,get_gaussian_kernel1d
import matplotlib.pyplot as plt
import random
import numpy as np
import torch
import torch.nn.functional as F
device = torch.device("cuda")
def CowMix(X,target,criterion):
  """
  Mezcla las imagenes dentro de un mismo lote usando un mascara de formas
  irregulares.        
  """
  B,_,H,W = X.shape

  #Proporcion de pixeles que se remplazan
  p = torch.rand(B,1,1,1,device=X.device)
  
  #Tamaño de las marcas de la mascara
  r = torch.randint(4,16,(1,),device=X.device)

  mask = torch.randn(B,1,r,r,device=X.device)
  mask = F.interpolate(mask,size=(H,W),mode="bilinear",align_corners=False)

  mean = mask.mean(dim=(1,2,3),keepdim=True)
  std = mask.std(dim=(1,2,3),keepdim=True)
  tao = mean+1.4*std*torch.erfinv(2*p-1)

  mask = mask>tao

  idx = torch.randperm(B)
  X = mask*X + torch.logical_not(mask)*X[idx]
  out = model(X)
  loss = (1-p)*criterion(out,target)+p*criterion(out,target[idx])
  return loss
