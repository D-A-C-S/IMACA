import torch
import torch.nn as nn
import numpy as np

__all__ = ['logistic_loss','MLP','NVPModel']

GC = np.log(2*np.pi)
def logistic_loss(h,logpz):
  
  zeros = torch.zeros_like(h)
  logpx = (  -torch.logsumexp(torch.stack([h,zeros],dim=-1),dim=-1) -
         torch.logsumexp(torch.stack([-h,zeros],dim=-1),dim=-1)   )
  
  #logpx = -0.5*(h**2+GC)
  logpx = logpx.sum(dim=1)#Variables independientes en espacio H
  logpx = logpz+logpx
  return -logpx

class MLP(nn.Module):
  def __init__(self,in_dim,out_dim,width,activation = False):
    super().__init__()
    
    self.relu = nn.ReLU()
    self.out_act = nn.Tanh() if activation else nn.Identity()
    self.linear1 = nn.Linear(in_dim,width)
    self.linear2 = nn.Linear(width,width)
    self.linear3 = nn.Linear(width,width)
    self.linear4 = nn.Linear(width,width)
    self.linear5 = nn.Linear(width,out_dim)
  def forward(self,x):
    out = self.relu(self.linear1(x))
    out = self.relu(self.linear2(out))
    out = self.relu(self.linear3(out))
    out = self.relu(self.linear4(out))
    out = 3*self.out_act(self.linear5(out))
    return out

class NVPModel(nn.Module):
  """
  Modelo generativo invertible
  forward: X-> H, det(dH/dX)
  inverse: H->X
  """
  def __init__(self,d=1,width=10):
    """
    args:
         d:mitad de la dimension total del vector de entrada
         width: numero de neuronas por capa oculta del MLP
    """
    super().__init__()

    self.s1 = MLP(d,d,width,activation=True)
    self.s2 = MLP(d,d,width,activation=True)
    self.s3 = MLP(d,d,width,activation=True)
    self.s4 = MLP(d,d,width,activation=True)

    self.t1 = MLP(d,d,width)
    self.t2 = MLP(d,d,width)
    self.t3 = MLP(d,d,width)
    self.t4 = MLP(d,d,width)
    self.d = d

  def forward(self,x):
    logph = 0
    #x1 = x[:,:self.d]
    #x2 = x[:,self.d:]
    B,D = x.shape
    
    x = x.view(B,self.d,2)
    x1 = x[:,:,0]
    x2 = x[:,:,1]

    h1 = x1
    scale = torch.exp(self.s1(h1))
    h2 = x2*scale + self.t1(h1)
    logph+=torch.log(scale).sum(dim=1)

    h2 = h2
    scale = torch.exp(self.s2(h2))
    h1 = h1*scale + self.t2(h2)
    logph+=torch.log(scale).sum(dim=1)

    h1 = h1
    scale = torch.exp(self.s3(h1))
    h2 = h2*scale + self.t3(h1)
    logph+=torch.log(scale).sum(dim=1)

    h2 = h2
    scale = torch.exp(self.s4(h2))
    h1 = h1*scale + self.t4(h2)
    logph+=torch.log(scale).sum(dim=1)

    #h = torch.cat([h1,h2],dim=1)
    h = torch.stack([h1,h2],dim=2).view(B,D)

    return h,logph
  
  def inverse(self,h):
    B,D = h.shape
    
    h = h.view(B,self.d,2)
    h1 = h[:,:,0]
    h2 = h[:,:,1]
    #h1 = h[:,:self.d]
    #h2 = h[:,self.d:]

    h2 = h2
    h1 = (h1-self.t4(h2))*torch.exp(-self.s4(h2))

    h1 = h1
    h2 = (h2-self.t3(h1))*torch.exp(-self.s3(h1))

    h2 = h2
    h1 = (h1-self.t2(h2))*torch.exp(-self.s2(h2))

    h1 = h1
    h2 = (h2-self.t1(h1))*torch.exp(-self.s1(h1))

    #x = torch.cat([h1,h2],dim=1)
    x = torch.stack([h1,h2],dim=2).view(B,D)
    return x
