import torch
import torch.nn as nn
import torch.nn.functional as F

from .ModelWrapper import InvertedModel
from .Parametrizaciones import (ColorDecorrelation,
                                BilateralFilter,
                                Normalization,
                                IFFT2,
                                SmoothCircles)
from .Transformaciones import (Jitter,
                               PadTensor,
                               ScaleTensor,
                               RotateTensor,
                               Standardization)


def ToNumpy(Tensor):
  return Tensor.detach().cpu().numpy().squeeze().transpose(1,2,0)

def interpolate_space(inp,scale_factor):
  return F.interpolate(inp,scale_factor=scale_factor,mode='bilinear',align_corners = True)

def interpolate_freq(inp,scale_factor):

  image = torch.irfft(inp,signal_ndim=2)
  _,_,S,_ = image.shape
  if S%2==0:
    image = image[:,:,:,:-1]
  image = interpolate_space(image,scale_factor)
  out = torch.rfft(image,signal_ndim=2)
  return out

transforms = [Standardization(),
              PadTensor(12),
              Jitter(8),
              RotateTensor(5),
              ScaleTensor(1.1),
              Jitter(4)
              ]


def MaximizeActivation(model,layer,canal=0,shape=120,LR=0.01,
                       iteraciones=100,octaves=8,neurona=False,alpha=0.03,
                       circles=None):
  device = "cuda"
  parameterization = [IFFT2(shape,alpha=alpha),nn.Sigmoid(),
                    ColorDecorrelation("imagenet"),Normalization()]
  
  if circles:
    parameterization = [SmoothCircles(shape,circles),Normalization()]

  modelo = InvertedModel(model,layer,shape=shape,
                        transforms=transforms,
                        parameterization=parameterization).to(device)

  optimizer = torch.optim.Adam(modelo.get_params(),lr=LR)#[0.01--0.25]#0.05
  scale_factor = 1.2
  for octave in range(octaves):
    if octave>0:
      modelo.input_parameter = nn.Parameter(interpolate_freq(modelo.input_parameter,scale_factor))
      if circles:
        modelo.input_parameter = nn.Parameter(interpolate_space(modelo.input_parameter,scale_factor))
      optimizer = torch.optim.Adam(modelo.get_params(),lr=LR)

    for i in range(iteraciones):

      modelo.zero_grad()
      out = modelo()
      
      if neurona:
        _,_,x,y = out["activation"].shape
        loss = -out["activation"][:,canal,x//2,y//2].mean()
      else:
        loss = -out["activation"][:,canal].mean()

      loss.backward()
      optimizer.step()
      if i%100==0:
        print(loss.item())

  print(out["activation"].shape)
  return out
