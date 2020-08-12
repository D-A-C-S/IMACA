import cv2
import torch
import numpy as np
import random
"""
Transformaciones para imagenes en formato ndarray
No se esta usando
"""
class Rescale(object):
  """Cambia el tamaño de la imagen,no altera la relacion de aspecto"""
  def __init__(self,output_size):
    
    assert isinstance(output_size, (int, tuple))
    self.output_size = output_size

  def __call__(self, image):

    x, y = image.shape[:2]
    if isinstance(self.output_size, int):
        if x > y:
            new_x, new_y = self.output_size * x / y, self.output_size
        else:
            new_x, new_y = self.output_size, self.output_size * y / x
    else:
        new_h, new_w = self.output_size

    cols, rows = int(new_x), int(new_y)

    img = cv2.resize(image, (rows, cols))
    
    return img

class RandomCrop(object):
  """Recorta la imagen en una posicion aleatoria, si el argumento de entrada es int,
     el corte es cuadrado"""
  def __init__(self,output_size):
    assert isinstance(output_size, (int, tuple))
    if isinstance(output_size, int):
        self.output_size = (output_size, output_size)
    else:
        assert len(output_size) == 2
        self.output_size = output_size
        
  def __call__(self,image):
    "Punto de referencia,esquina superior izquierda"
    h, w = image.shape[:2]
    new_h, new_w = self.output_size

    top = np.random.randint(0, h - new_h)
    left = np.random.randint(0, w - new_w)
    
    return image[top:top+new_h,left:left+new_w]

class CenterCrop(object):
  """Recorta la imagen en la posicion central"""
  def __init__(self,output_size):
    """args: 
      output_size:  (int)tamaño de la imagen cuadrada de salida
    """
    assert isinstance(output_size,int)
    self.output_size = output_size
    
  def __call__(self,imagen):
    "imagen : numpy array en formato H,W,C"
    h,w,_ = imagen.shape
    
    top = int((h-self.output_size)/2)
    left = int((w-self.output_size)/2)
    
    return imagen[top:top+self.output_size,left:left+self.output_size]
    
    
class FlipRotate(object):
  """Rotaciones de 90-180-270° y reflexion de imagen aleatoria"""
  def __call__(self,image):
    if random.randint(0,1):
      image=np.flipud(image)
    return (np.rot90(image,k=random.randint(0,3))).copy()
  
class ToTensor(object):
  """Cambio de formato de uint8 array a Tensor de imagen"""
  def __call__(self, image):
      # swap color axis because
      # numpy image: H x W x C
      # torch image: C X H X W
      return torch.from_numpy(image.transpose((2, 0, 1))).float()/255
