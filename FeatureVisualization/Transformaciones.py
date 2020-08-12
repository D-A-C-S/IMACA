import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize


"""
Transformaciones diferenciables de tensores,
que se usan como regularizadores para mejorar la
estructura y reducir las altas frecuencias de las
imagenes generadas por maximizacion de activacion
de neuronas o canales
"""
class Jitter(nn.Module):
  """Recorta el tensor que representa una imagen sobre
     una ventana aleatoria"""
  def __init__(self,max_jitter):
    "max_jitter:numero de pixeles que pierde cada dimension espacial"
    super().__init__()
    self.double_max_shift = max_jitter
  
  def forward(self,x):
    _,_,shape,_ = x.shape
    new_shape = shape-self.double_max_shift
    h_min = random.randint(0,self.double_max_shift)
    w_min = random.randint(0,self.double_max_shift)
    return x[:,:,h_min:h_min+new_shape,w_min:w_min+new_shape]
###
class PadTensor(nn.Module):
  "Tensor (B,C,H,W)"
  def __init__(self,pad,value=0):
    """
    args:
        pad:numero de pixeles que se añaden a cada extremo de las
            dimensiones espaciales
    """
    super().__init__()
    self.pad = pad
    self.value = value
  def forward(self,x):
    return F.pad(x,(self.pad,self.pad,self.pad,self.pad),value=self.value)
###
class ScaleTensor(nn.Module):
  """
     Interpola las dimensiones espaciales de un tensor
     que representa un lote de imagenes (B,C,H,W).
     La escala de interpolacion se selecciona aleatoriamente,
     en un rango predefinido.
  """
  def __init__(self,max_scale,interpolate ="bilinear"):
    """
    args:
        max_scale: rango de la escala de interpolacion [-max_scale,max_scale]
        "interpolation:metodo de... uno de ['bilinear','bicubic','nearest']
    """
    assert max_scale>=1.0 and max_scale<=2.0
    super().__init__()
    self.max_scale = max_scale
    self.min_scale = 1/max_scale
    self.mode = interpolate

  
  def forward(self,x):

    scale_factor = random.uniform(self.min_scale,self.max_scale)
    out = F.interpolate(x,scale_factor=scale_factor,
                          mode=self.mode,align_corners=True)

    return out
###
class RotateTensor(nn.Module):
  """
     Rotacion aleatoria en un rango predefinido 
     de un tensor que representa una imagen.(B,C,H,W)
  """
  def __init__(self,max_deg,interpolate="nearest"):
    """
    args:
        max_deg:maxima rotacion de la imagen en grados(degrees)
        interpolate: metodo de interpolacion, uno de ['nearest','bilinear']
    """
    super().__init__()
    self.register_buffer("rotation_matrix",torch.randn(1,2,3))
    self.max_rads = np.deg2rad(max_deg)
    self.interpolate = interpolate

  def _get_rotation_matrix(self,rads):
    c = np.cos(rads)
    s = np.sin(rads)
    out = torch.tensor([[c,-s,0],[s, c,0]],
           device=self.rotation_matrix.device,
           dtype=torch.float32).unsqueeze(0)
    return out
    
  def forward(self,x):
    rads = random.uniform(-self.max_rads,self.max_rads)
    self.rotation_matrix = self._get_rotation_matrix(rads)    
    Grid = F.affine_grid(self.rotation_matrix,x.shape,align_corners=True)
    
    out = F.grid_sample(x,Grid,mode=self.interpolate,align_corners=True)
    return out
###

class Standardization(nn.Module):
  "!Esta operacion asume tensor (1,3,H,W)"
  def __init__(self):
    super().__init__()
    self.normalize = Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
  def forward(self,x):
    out = x-x.min()
    out = x/x.max()
    out = self.normalize(out.squeeze())
    return out.unsqueeze(0)

Lucid_transforms = [
              PadTensor(12),
              Jitter(8),
              RotateTensor(5),
              ScaleTensor(1.1),
              Jitter(4)
              ]

