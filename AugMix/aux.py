from PIL import ImageDraw
from PIL import Image

from torchvision.transforms import ToPILImage
from albumentations.augmentations.transforms import GridDistortion,ElasticTransform
import numpy as np
from random import uniform

import torch
import torch.nn.functional as F


class CircularMask(object):
  def __init__(self,size=256):
    self.mask = Image.new("L", (size,size), 0)
    self.background = Image.new("RGB",(size,size),0)
    draw = ImageDraw.Draw(self.mask)
    draw.ellipse((0, 0, size, size), fill=255)
  
  def __call__(self,image):
    out = Image.composite(image,self.background,self.mask)
    return out

class GridTransforms(object):
  def __init__(self):
    self.transform1 = ElasticTransform()
    self.transform2 = GridDistortion()
    self.topil = ToPILImage()
  def __call__(self,pil_image):
    image = np.array(pil_image)
    if uniform(0.0,1.0)>0.5:
      image = self.transform1(image=image)["image"]
    else:
      image = self.transform2(image=image)["image"]
      
    image = self.topil(image)
    return image


def CowMix(X,target,model):
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
  p = p.squeeze()
  loss = ( (1-p)*F.cross_entropy(out,target,reduction='none') +
               p*F.cross_entropy(out,target[idx],reduction='none') )
  return loss.mean() , out
