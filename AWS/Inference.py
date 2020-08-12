#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 19:13:10 2019

@author: lejo
"""
#Pendiente softmax!
import onnxruntime
import numpy as np
from PIL import Image
from preprocess import ToNumpy,Resize,CenterCrop,Normalize,Compose


ESPECIES = ['Cedrelinga cateniformis','Cedrela odorata'        ,'Humiriastrum procerum',
            'Dialyanthera gracilipes','Eucalyptus globulus'    ,'Handroanthus chrysanthus',
            'Cordia alliodora'       ,'Campnosperma panamensis','Fraxinus uhdei'           ]
ESPECIESC = ["Achapo","CedroCosteño","Chanul","Cuangare","EucaliptoBlanco",
           "GuayacanAmarillo","NogalCafetero","Sajo","Urapan"]
BASE_URL = "https://idemadxreference.s3.amazonaws.com/"
IMAGENES = [BASE_URL+especie+".jpg" for especie in ESPECIESC]
RSIZE = 256
RCROP = 240

device = "cpu"

class LoadedModel():
  def __init__(self,model_path):

    self.modelo = onnxruntime.InferenceSession(model_path)
    self.transforms = self.get_transforms()
    self.CodToEspecie = ESPECIES    
    
  def get_transforms(self):
    normalize = Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
    
    inference_transforms = Compose([Resize(RSIZE),
                             CenterCrop(RCROP),
                             ToNumpy,
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
    Image = self.transforms(image)[np.newaxis,...]
    out,in_dist = self.modelo.run([],{'input':Image})
    if not in_dist:
      respuesta = "No se encontraron coincidencias" 
      return respuesta,False
      
    #out= softmax(out,dim=1)
    indices = np.argsort(out[0])[-3:][::-1]
    values = out[0,indices]
    ###
    respuestas=[]
    urls = []
    for value,idx in zip(values,indices):
      respuestas.append(
        '_{}_'.format(self.CodToEspecie[idx]))
      urls.append(IMAGENES[idx])
    
    return respuestas,urls




  
