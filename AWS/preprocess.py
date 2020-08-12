import numpy as np

__all__ = ['CenterCrop','Resize','ToNumpy','Normalize','Compose']

class CenterCrop(object):
  "Recorte rectangular de una imagen PIL"
  def __init__(self,H,W=None):
    if not W:
      W = H
    self.W = W
    self.H = H    
  def __call__(self,image):

    if self.H>image.height or self.W>image.width:
      raise ValueError("La imagen es muy pequeña para recortar")

    #(0,0) upper left corner
    left = (image.width-self.W)//2
    right = left+self.W

    upper = (image.height-self.H)//2
    lower = upper+self.H

    out = image.crop(box=(left,upper,right,lower))
    return out

class Resize(object):
  """
  Redimensiona una imagen PIL 
  conservando la relacion de aspecto
  """
  def __init__(self,size):
    self.size = size
    self.resample = 2#Bilinear
  def __call__(self,image):
    H,W = image.height,image.width

    if H>W:
      tH = int(H*self.size/W)
      tW = self.size
    else:
      tH = self.size
      tW = int(W*self.size/H) 
    out = image.resize((tW,tH),resample=self.resample)
    return out

def ToNumpy(image):
  """
  Convierte una imagen RGB uint8
  de PIL a Numpy...!png
  """
  out = np.array(image,dtype=np.float32)/255
  out = out.transpose(2,0,1)
  return out

class Normalize(object):
  """
  Opera sobre numpy arrays (C,H,W)
  """
  def __init__(self,mean,std):
    self.mean = np.array(mean,dtype=np.float32).reshape(-1,1,1)
    self.std = np.array(std,dtype=np.float32).reshape(-1,1,1)
  
  def __call__(self,image):
    out = (image-self.mean)/self.std
    return out.astype(np.float32)

class Compose(object):
  def __init__(self,transforms):
    self.transforms = transforms

  def __call__(self,inp):
    out = inp
    for transform in self.transforms:
      out = transform(out)
    return out