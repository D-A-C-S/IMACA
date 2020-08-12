import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from .SinUso.Transformaciones import Rescale

from os import scandir,environ
from os import sep as osSep#caracter separador del sistema operativo
from os.path import getmtime

__all__=['list_recursive_path','get_df_from_path','EMFechas',
         'ToImage,DeNormalize',"estimate_orientation"]
         
environ['TZ'] = "EST"
time.tzset()

def list_recursive_path(Lista,path,formato):
  with scandir(path) as pantera:#scandirIterator
    for obj in pantera:#DirEntry
      if obj.is_dir():
        list_recursive_path(Lista,obj.path,formato)
      elif obj.name[-3:]==formato:
        Lista.append(obj.path)
    return      

def get_df_from_path(path,nombre_cols = None,depth=2,formatos=['jpg','png','bmp']):
  """
  Construye un dataframe a partir de la estructura de las carpetas
  depth : numero de columnas con el nombre de las carpetas.
  					Se generan dos columnas adicionales,el nombre del archivo y el directorio completo.
  nombre_cols = renombra las columnas 
  formato : filtra archivos por su extension
  """
  
  Lista=[]
  for formato in formatos:
    list_recursive_path(Lista,path,formato)
    
  Tlista = []
  Nsplits = depth+1
  for path in Lista:
    Tlista.append(path.split(osSep)[-1*Nsplits:]+[path])
    assert len(Tlista[-1])==depth+2
    
  df=pd.DataFrame(Tlista)
  if nombre_cols:
  	assert len(nombre_cols)==depth
  	col_dict = {i:ncol for i,ncol in enumerate(nombre_cols)}
  	df.rename(columns=col_dict,inplace=True)
  df.rename(columns={depth:'filename',depth+1:'path'},inplace=True)
  return df

class EMFechas:
  "Almacena funciones para la extraccion y operacion con fechas"

  @classmethod
  def extraer_fecha(self,filename,path):
    """Obtiene la fecha del nombre del archivo,y el grupo de origen(camara)
       si no es posible reporta la ultima fecha de modificacion"""
    if filename[:3]=="IPC":#Formato de USBCamera:IPC_2019-09-09.15.56.13.6770.jpg
      struct_time = time.strptime(filename[:23],"IPC_%Y-%m-%d.%H.%M.%S")
      return time.strftime("%Y-%m-%d %H:%M:%S",struct_time),0

    elif filename[0].isdigit():#Formato CameraFi:190813_155201.png
      struct_time = time.strptime(filename[:13],"%y%m%d_%H%M%S")
      return time.strftime("%Y-%m-%d %H:%M:%S",struct_time),0
    
    elif filename[:3] in ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]: #Formato AIMP Camera: Fri Aug 30 14-20-37.jpg
      struct_time = time.strptime(filename[:19]+"2019", "%a %b %d %H-%M-%S%Y")
      return time.strftime("%Y-%m-%d %H:%M:%S",struct_time),1
    else:
      struct_time = time.localtime(getmtime(path))
      return time.strftime("%Y-%m-%d %H:%M:%S",struct_time),1

  @classmethod
  def diferencia(self,time1,time2):
    """
    Obtiene el tiempo en segundos transcurrido(en valor absoluto) entre horas.
    La maxima diferencia que se reporta es de un dia(86400 segundos)

    time:tuplas de la forma struct_time:
    EJ: time.struct_time(tm_year=2000, tm_mon=11,tm_mday=30, tm_hour=0,
                         tm_min=0, tm_sec=0,tm_wday=3, tm_yday=335, tm_isdst=-1)
    """
    if time1[:3]!=time2[:3]:
      return 84600
    
    h1,m1,s1 = time1[3:6]
    h2,m2,s2 = time2[3:6]

    S1 = h1*3600+m1*60+s1
    S2 = h2*3600+m2*60+s2
    return abs(S1-S2)

import numpy as np
def ToImage(Image):
  Image = np.array(Image.detach().cpu())
  Image = Image.transpose(1,2,0)
  return Image

def DeNormalize(tensor):
  mean = [0.485, 0.456, 0.406]
  std = [0.229, 0.224, 0.225]
  tensor = tensor*torch.tensor(std).unsqueeze(1).unsqueeze(2)
  tensor = tensor+torch.tensor(mean).unsqueeze(1).unsqueeze(2)
  return tensor

def estimate_orientation(image_path,draw=False):
  fld = cv2.ximgproc.createFastLineDetector()

  image = cv2.imread(image_path)
  image = Rescale(256)(image)
  gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
  lines = fld.detect(gray)
  if lines is None:
    print("No se encontraron lineas en:"+image_path)
    return 0
  x = lines[...,0]-lines[...,2]
  y = lines[...,1]-lines[...,3]
  _,angulos = cv2.cartToPolar(x,y,angleInDegrees=True)
  count,edges = np.histogram(angulos,bins=50,range=(0,180))
  max_pos = np.argmax(count)
  angulo = (edges[max_pos]+edges[max_pos+1])/2

  if draw:
    line_on_image = fld.drawSegments(image, lines)
    plt.imshow(line_on_image)
      
  return angulo-90


def applyMap(mapa,cmap=2):
  "mapa:Imagen (H,W)"
  mapa = np.array(mapa)
  mapa = mapa-mapa.min()
  mapa = mapa/mapa.max()
  mapa = 255-mapa*255
  mapa = mapa.astype(np.uint8)
  mapa = cv2.applyColorMap(mapa,2)
  return mapa/255.0

def saveImage(filename,image):
  image = cv2.cvtColor(image,4)
  image = np.clip(image,0.0,1.0)
  image = (image*255).astype(np.uint8)
  cv2.imwrite(filename,image)
