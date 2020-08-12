#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 13:29:38 2019

@author: lejo
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from .Parametrizaciones import IFFT2


  
class InvertedModel(nn.Module):
  """
  Habilita la diferenciacion con respecto a la entrada,
  mantiene los parametros de el modelo base intactos.
  """
  def __init__(self,modelo,modulo,transforms=[],parameterization = [],shape=224,std=0.01):
    """
      Nota:Mantener el modelo en modo evaluacion
      transforms: lista de operaciones diferenciables definidas en terminos de tensores.
      parameterization : lista de parametrizaciones diferenciables de la imagen.
    """
    super().__init__()
    self.modelo = deepcopy(modelo)
    self.modelo.eval()
    modulo = self.get_module(modulo)

    self._activation = None
    self.handle= modulo.register_forward_hook(self.gancho)
    
    self.transforms = nn.ModuleList(transforms)
    self.parameterization = nn.ModuleList(parameterization)
        
    ifft = any([isinstance(para,IFFT2) for para in parameterization])
    S = shape
    if ifft:
      self.input_parameter = nn.Parameter(std*torch.randn(1,3,S,int(S/2)+1,2))
    else:
      self.input_parameter = nn.Parameter(std*torch.randn(1,3,S,S))

  def forward(self,x=None):
    "Opera sobre el input_parameter,no tiene entradas externas"
    image = self.input_parameter
    for transform in self.parameterization:
      image = transform(image)

    out = image
    for transform in self.transforms:
      out = transform(out)
    
    if isinstance(x,torch.Tensor):
      out = x

    logits = self.modelo(out)
    return {"logits":logits,"activation":self._activation,"image":image}

  def get_params(self):
    return [self.input_parameter]+list(self.parameterization.parameters())

  
  def gancho(self,module,inputs,outputs):
    self._activation = outputs

  def __del__(self):
    self.handle.remove()

  def get_module(self,modulo):
    for name,module in self.modelo.named_modules():
      if name==modulo:
        return module
    raise ValueError('No se encontro el modulo')
                 




