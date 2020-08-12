import torch
import torch.nn as nn
import torch.nn.functional as F

from ..OODDeteccion import InvertedModel

def get_activations(modelo,layer,dataloader,device="cuda"):
  ids = []
  all_activations = []
  hooker = InvertedModel(modelo,layer)
  with torch.no_grad():
    for sample in dataloader:
      image = sample["image"].to(device)
      ids += sample["id"]
      logits,activations= hooker(image)
      all_activations.append(activations)
    all_activations = torch.cat(all_activations)
  return all_activations,ids

class GradCam():
  """
  Genera un mapa de activaciones de la clase de interes para una imagen,
  de acuerdo a la importancia de las caracteristicas de una capa 
  convolucional(Usualmente la ultima)
  """
  def __init__(self,modelo,modulo,clase=0,interpolate=True):
    self.modelo = modelo
    self.modelo.eval()
    self.interpolate = interpolate

    self._activation = None

    self.handle= modulo.register_forward_hook(self.gancho)
    self.grad_handle = None

    self.avg_pool = nn.AdaptiveAvgPool2d(1)
    self.clase = clase

  def __call__(self,x):
    
    out = self.modelo(x)
    mascara = torch.zeros_like(out)
    mascara[:,self.clase]+=1 
    out = out*mascara
    loss = out.sum()

    self.grad_handle = self._activation.register_hook(self.grad_hook)
    loss.backward()
    
    with torch.no_grad():
      act_map = self._activation*self.avg_pool(self._activation.grad)
      act_map = act_map.sum(dim=1,keepdim=True)
      act_map = F.relu(act_map)
      if self.interpolate:
        _,_,S,_ = x.shape
        act_map = F.interpolate(act_map,size=S,mode='bicubic',align_corners = True)
    return act_map
        
  def grad_hook(self,grad):
    self._activation.grad = grad

  def gancho(self,module,inputs,outputs):
    self._activation = outputs

  def __del__(self):
    self.handle.remove()
    self.grad_handle.remove()

