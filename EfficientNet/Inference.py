#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 19:13:10 2019

@author: lejo
"""

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToTensor,Resize,CenterCrop,Normalize
import efficientnet_builder
from torch.nn.functional import softmax
from os import linesep

ESPECIES = ['Eucalipto blanco', 'Cedro costeño', 'Cuangare', 'Achapo', 'Guayacan amarillo', 'Urapan', 'Chanul', 'Nogal cafetero', 'Sajo']
ESPECIES.sort()
RSIZE = 256
RCROP = 224
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class LoadedModel():
  def __init__(self,jit_model):

    self.model = torch.jit.load(jit_model,map_location=device)
    self.transforms = self.get_transforms(res)
    self.CodToEspecie = ESPECIES    
    
  def get_transforms(self):
    normalize = Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
    
    inference_transforms = transforms.Compose([Resize(RSIZE),
                             CenterCrop(RCROP),
                             ToTensor(),
                             normalize
                              ])
    return inference_transforms
  
  @staticmethod
  def load_image(path):
  with open(path, 'rb') as f:
      img = Image.open(f)
      return img.convert('RGB')
    
  
  def predict(self,image_filename):
    image = self.load_image(image_filename)
    Image = self.transforms(image).unsqueeze(0).to(device)
    out = self.modelo(Image)
    out = softmax(out,dim=1)
    values,indices = out.detach().cpu().squeeze().topk(3)  
    ###
    respuesta=""
    for value,idx in zip(values,indices):
      respuesta+='{}: {:.3f}%'.format(self.CodToEspecie[idx],value*100)+linesep
    return respuesta




  
