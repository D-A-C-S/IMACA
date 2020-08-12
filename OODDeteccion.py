import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

__all__ = ['InvertedModel','MahalanobisDistance','MixedModel']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class InvertedModel(nn.Module):
  """
  Envuelve un modelo para capturar el vector 
  de activacion de una capa de interes
  """
  def __init__(self,modelo,modulo):
    """
    modulo: nombre del modulo
    """
    super().__init__()
    
    self.modelo = deepcopy(modelo)
    self.modelo.eval()
    modulo = self.get_module(modulo)

    self._activation = None
    self.handle= modulo.register_forward_hook(self.gancho)
    
  def forward(self,x):
    logits = self.modelo(x)
    return logits,self._activation
  
  def gancho(self,module,inputs,outputs):
    self._activation = outputs

  def get_module(self,modulo):
    for name,module in self.modelo.named_modules():
      if name==modulo:
        return module
    raise ValueError('No se encontro el modulo')

class MahalanobisDistance(nn.Module):
  """
  Calcula la distancia de un vector al centroide mas cercano
  de una mezcla de distribuciones gaussianas que comparten la
  misma matriz de covarianza.
  args:
      means[D,C]: promedio por clase de las variables aleatorias
      covar[D,D]: covarianza de las variables aleatorias
  """
  def __init__(self,means,covar):
    super().__init__()
    self.register_buffer("means",means.unsqueeze(0))#1,D,C
    self.register_buffer("covar",covar.unsqueeze(0))#1,D,D
    self.register_buffer("alpha",torch.inverse(self.covar))

  def forward(self,X):#[B,D]
    d2 = X.unsqueeze(2)-self.means#B,D,C
    out = (d2.transpose(1,2)@self.alpha@d2).diagonal(dim1=1,dim2=2)#B,C,C->B,C
    out,_ = out.min(dim=1)#B

    return out

class MixedModel(nn.Module):
  """
  Ajusta una distribucion gaussiana con covarianzas compartidas,
  a los vectores de activacion de una capa de interes.

  """
  def __init__(self,model,layer,dataloader):
    super().__init__()
    #Hook model
    self.model = InvertedModel(model,layer)
    
    #Ajuste de gaussiana
    features,labels,_ = self.get_features(dataloader)
    means,covar = self.TiedGaussianMixture(features,labels)
    
    #Definicion de operador para calculo de distancia.
    self.mdistance = MahalanobisDistance(means.T,covar)
    
    self.criterio = 0

  def forward(self,x):
    logits,features = self.model(x)
    if len(features.shape)==4:
      features = features.mean(dim=(2,3))
    distancia = self.predict(features=features)
    in_distribution =  torch.le(distancia,self.criterio).squeeze()
    return logits,in_distribution

  def get_features(self,dataloader):
    features = []
    labels = []
    indices = []
    with torch.no_grad():
      for sample in dataloader:
        if isinstance(sample,dict):
          image = sample["image"]
          label = sample["label"]
          indices+=sample["id"]
        else:
          image,label = sample
        image = image.to(device)
        _,activation = self.model(image)
        if len(activation.shape)==4:
          activation = activation.mean(dim=(2,3))
        features.append(activation)
        labels.append(label)

      Features = torch.cat(features,dim=0)
      Labels = torch.cat(labels)
    return Features,Labels,indices
    
  @staticmethod
  def TiedGaussianMixture(X,Y):
    "out: means[C,D],covar[D,D]"
    X = X.detach().cpu().numpy()
    Y = Y.detach().cpu().numpy()
    N,D = X.shape
    classes = np.unique(Y)
    Gmeans = []
    cum_covar = np.zeros((D,D))
    for i in classes:
      subset = X[Y==i]
      M = len(subset)

      Gmeans.append(subset.mean(axis=0))
      cum_covar += (M/N)*np.cov(subset,rowvar=False,bias=False)
    Gmeans = np.stack(Gmeans)
    return torch.Tensor(Gmeans).to(device),torch.Tensor(cum_covar).to(device)

  def predict(self,dataloader=None,features=None):
    if dataloader:
      features,_,_ = self.get_features(dataloader)

    with torch.no_grad():
      distancia = self.mdistance(features)
    return distancia