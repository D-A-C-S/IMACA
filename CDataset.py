import torch
from torch.utils.data import Dataset
from PIL import Image

import random
import cv2
import numpy as np

def pil_loader(path):
  #De pytorch ImageFolder Dataset
  # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
  with open(path, 'rb') as f:
      img = Image.open(f)
      return img.convert('RGB')

class EspeciesMaderablesDataset(Dataset):

  def __init__(self, dataframe, transform=None,style_transform=None):
    """
    dataframe: columnas obligatorias:especie,filename,path
    """
    self.df = dataframe.copy()
    self.transform = transform
    self.style_transform = style_transform
    self.stylize = "style_path" in self.df.columns

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    df_sample = self.df.iloc[idx]
    image = pil_loader(df_sample.path)
    label = torch.tensor(df_sample.label,dtype=torch.long)

    if self.transform:
        image = self.transform(image) 

    out = {'image': image, 'label': label,
            'id':str(df_sample.name)}
    
    if self.stylize:
      style = pil_loader(df_sample.style_path)
      if self.style_transform:
        style = self.style_transform(style)
      out["style"] = style

    return out

class BasicDataset(EspeciesMaderablesDataset):

  def __getitem__(self, idx):
    df_sample = self.df.iloc[idx]
    image = pil_loader(df_sample.path)
    label = torch.tensor(df_sample.label,dtype=torch.long)
    if self.transform:
        image = self.transform(image) 
    return image,label


class SyntheticWood():
  def __init__(self):
    self.IMSIZE = 256
    self.set_default()
    self.set_species()

    if 2*max(self.PORE_RADIUS)>min(self.INTER_RAY_WIDTH):
      raise ValueError("El espacio entre radios debe ser mayor al tamaño de los poros")
  
  def set_species(self):
    "Sobre-escribir parametros de set_default"
    raise NotImplementedError
  
  def set_default(self):
    self.name = "Default"
    self.RAY_WIDTH = (2,4)
    self.INTER_RAY_WIDTH = (10,12)
    self.PORE_RADIUS = (3,4)
    self.NUM_PORES = (0,15)
    self.NOISE_STD = 25
    self.NOISE_MEAN = 0
    self.PORE_THICKNESS = cv2.FILLED


  def generate(self):
    total_poros = 0
    #Fibras
    base = 100*np.ones((self.IMSIZE,self.IMSIZE),dtype=np.float)
    posicion = 0
    while posicion<self.IMSIZE:
      #Rayo
      width = random.randint(*self.RAY_WIDTH)
      imagen = cv2.line(base,(posicion,0),(posicion,250),color=(250,),thickness=width)
      posicion+=width
      if posicion+width>self.IMSIZE:
        break
      #Poros
      width = random.randint(*self.INTER_RAY_WIDTH) 
      poros_del_rayo = random.randint(self.NUM_PORES[0],self.NUM_PORES[1])
      total_poros+=poros_del_rayo
      for i in range(poros_del_rayo):
        radio = random.randint(*self.PORE_RADIUS)
        centro = (random.randint(posicion+radio,posicion+width-radio),random.randint(0,self.IMSIZE))
        
        color=50
        if self.PORE_THICKNESS>0:
          imagen = cv2.circle(base,centro,radio,color=(200,),
                              thickness=cv2.FILLED)
          color = random.choice([50,150])
        imagen = cv2.circle(base,centro,radio-self.PORE_THICKNESS,
                            color=(color,),thickness=cv2.FILLED)
      posicion+=width

    imagen = cv2.blur(imagen,(3,3))
    imagen+= np.random.randn(self.IMSIZE,self.IMSIZE)*self.NOISE_STD+self.NOISE_MEAN
    imagen = np.stack([imagen,imagen,imagen],axis=2)/255.0
    imagen = np.clip(imagen.astype(np.float32),0.0,1.0)

    out = {"image":imagen,"total_poros":total_poros}
    return out

class Sajo(SyntheticWood):
  def set_species(self):
    self.name = "Sajo"
    self.RAY_WIDTH = (2,4)
    self.INTER_RAY_WIDTH = (10,12)
    self.PORE_RADIUS = (3,4)
    self.NUM_PORES = (5,15)
    self.NOISE_STD = 25
    self.NOISE_MEAN = 0

class SyntheticDataset(Dataset):

  def __init__(self, df,especies, transform=None,style_transform=None):
    """
    especies:lista de objetos con metodo para generar una imagen rgb(0,1), 
    y con el atributo name que debe coincidir con las especies existentes.
    """

    self.dfs = {especie.name:df[df.especie==especie.name] for especie in especies}
    self.especies = especies
    self.transform = transform
    self.style_transform = style_transform

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    especie = random.choice(self.especies)
    df = self.dfs[especie.name]

    synt = especie.generate()#RGB[0.0,1.0]
    image = synt["image"]
    df_sample = df.sample().iloc[0]
    label = torch.tensor(df_sample.label,dtype=torch.long)

    if self.transform:
        image = self.transform(image) 

    out = {'image': image, 'label': label,
            'id':str(df_sample.name)}
    
    if self.style_transform:
      style = pil_loader(df_sample.path)
      style = style.rotate(df_sample.Orientacion,2)#BILINEAR
      style = self.style_transform(style)
      out["style"] = style
      out["total_poros"] = synt["total_poros"]

    return out



  
    
      
      
