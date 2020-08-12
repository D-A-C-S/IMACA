"""
Operaciones diferenciables sobre tensores que 
deforman el espacio de la funcion que se optimiza
sin alterar el minimo global.
Aumenta la pendiente en direcciones preferenciales,
puede disminuir el tiempo de convergencia o darle caracteristicas
particulares a la solucion que se busca
"""
import numpy as np
import torch
import torch.nn as nn


class _SmoothCircle(nn.Module):
  "Dibuja un circulo"
  def __init__(self,width):
    super().__init__()
    self.mean = nn.Parameter(0.5*torch.randn(2,device="cuda"))
    self.sigma = nn.Parameter(0.1+0.1*torch.rand(1,device="cuda"))
    self.color = nn.Parameter(torch.rand(3,device="cuda").view(3,1,1))
    self.width  = width
    X,Y = torch.meshgrid(torch.linspace(-1,1,self.width),
                        torch.linspace(-1,1,self.width))
    self.register_buffer("X",X)
    self.register_buffer("Y",Y)

  def forward(self):

    d2 = ((self.X-self.mean[0])/self.sigma)**2+((self.Y-self.mean[1])/self.sigma)**2
    z = (1/6/self.sigma**2)*torch.exp(-d2)
    z = z.unsqueeze(0)*self.color
    return z

class SmoothCircles(nn.Module):
  "Dibuja circulos"
  def __init__(self,width,n=5):
    super().__init__()
    self.circles = nn.ModuleList([_SmoothCircle(width) for i in range(n)])
  def forward(self,*args):
    inp = torch.stack( [circle() for circle in self.circles] )
    inp = (1-inp.mean(dim=0)).unsqueeze(0)
    return inp#


class ColorDecorrelation(nn.Module):
  """Supone una imagen de entrada con colores no correlacionados, 
  y los transforma a un espacio correlacionado, de acuerdo a la distribucion
  que origino el modelo."""
  def __init__(self,covariance=None):
    super().__init__()

    if isinstance(covariance,np.ndarray):
      covariance_decomposition = self.get_decomposition(covariance)
    else:
      covariance_decomposition = self.imagenet_svd_sqrt()

    self.register_buffer("covariance_decomposition",torch.tensor(covariance_decomposition))

  @staticmethod
  def imagenet_svd_sqrt():
    #De Lucid:Feature Visualization:Github
    color_correlation_svd_sqrt = np.asarray([[0.26, 0.09, 0.02],
                                         [0.27, 0.00, -0.05],
                                         [0.27, -0.09, 0.03]]).astype("float32")
    max_norm_svd_sqrt = np.max(np.linalg.norm(color_correlation_svd_sqrt, axis=0))
    color_correlation_normalized = color_correlation_svd_sqrt / max_norm_svd_sqrt
    return color_correlation_normalized.T

  @staticmethod
  def get_decomposition(covariance):
    W = np.linalg.cholesky(np.linalg.inv(covariance))
    return np.linalg.inv(W)

  def forward(self,x):
    #x.shape 1,3,s1,s2
    #covariance.shape : 3,3

    x = x.transpose(1,3)#1,s2,s1,3
    out = x@self.covariance_decomposition
    out = out.transpose(1,3)
    return out
###

class BilateralFilter(nn.Module):
  """
  Filtro bilateral sobre Tensores que representan imagenes
  Preserva bordes,consiste en un promedio ponderado que da mayor 
  peso a pixeles cercanos,y pixeles similares en intensidad
  """
  def __init__(self,window_size=5,sigmaXY=1,sigmaZ=1):
    """
    argumentos:
    window_size = numero de pixeles de un lado de la ventana cuadrada de operacion.
    sigmaXY = modulacion de la contribucion espacial del filtro
    sigmaZ = modulacion de la contribucion de pixeles de intensidades cercanas
    """
    super().__init__()
    self.Wsize = window_size
    self.L = window_size**2
    self.sigmaXY2 = sigmaXY**2
    self.sigmaZ2 = sigmaZ**2
    self.pad = int(self.Wsize//2)
    self.desenrrollar = torch.nn.Unfold((self.Wsize,self.Wsize),padding = (self.pad,self.pad))
    self.register_buffer("similarityXY",self._build_spatial_kernel())


  def _build_spatial_kernel(self):
    """
    Construye un vector del tamaño de la ventana,
    que mide la distancia al cuadrado con respecto al pixel central.
    """
    xv,yv = np.meshgrid(range(-self.pad,self.pad+1),range(-self.pad,self.pad+1))
    distancia2 = xv**2+yv**2
    distancia2 = distancia2.ravel().reshape(1,1,-1,1).astype(np.float32)
    distancia2 = torch.from_numpy(distancia2)
    return self.gaussian(distancia2,self.sigmaXY2)

  @staticmethod
  def gaussian(X2,sigma2):
    return torch.exp(-0.5*X2/sigma2)

  def forward(self,X):
    
    n,C,H,W = X.shape
    un_X = self.desenrrollar(X)         #shape (n,C*L,H*W)
    un_X = un_X.view(n,C,self.L,H*W)    #shape (n,C,L,H*W)
    central_points = X.view(n,C,1,H*W)  #shape (n,C,1,H*W)

    similarityZ = (un_X-central_points)**2
    similarityZ = self.gaussian(similarityZ,self.sigmaZ2) #shape (n,C,L,H*W)
	
    similarity = similarityZ*self.similarityXY
    out = (similarity*un_X).sum(dim=2)
    out = (out)/similarity.sum(dim=2)#Normalizacion
    
    return out.view(n,C,H,W)


###
class Normalization(nn.Module):
  def forward(self,x):
    out = x-x.min()
    out = x/x.max()
    return out

###

class IFFT2(nn.Module):
  def __init__(self,w,alpha=0.03):
    super().__init__()
    self.alpha = alpha
    scale = torch.Tensor(self.get_scale(w))
    self.register_buffer('scale',scale)
  @staticmethod
  def rfft2d_freqs(h, w):#TOMADO DE LUCID:GIT!
    """Computes 2D spectrum frequencies."""

    fy = np.fft.fftfreq(h)[:, None]
    # when we have an odd input dimension we need to keep one additional
    # frequency and later cut off 1 pixel
    if w % 2 == 1:
        fx = np.fft.fftfreq(w)[: w // 2 + 1]
    else:
        fx = np.fft.fftfreq(w)[: w // 2 + 1]
    return np.sqrt(fx * fx + fy * fy)
  
  def get_scale(self,w):
    h = w
    freqs = self.rfft2d_freqs(h, w)
    scale = 1.0 / np.maximum(freqs, 1.0 / max(w, h))
    scale *= np.sqrt(w * h)
    scale = scale.reshape(1,1,w,-1,1)
    return scale
  
  def reset_scale(self,w):
    scale = self.get_scale(w)
    self.scale = torch.tensor(scale,device=self.scale.device,dtype=self.scale.dtype)   
  
  def forward(self,x):
    #x.shape B,C,H,W,2
    if x.shape[2]!=self.scale.shape[2]:
      self.reset_scale(x.shape[2])
    x = x*self.scale
    return self.alpha*torch.irfft(x,signal_ndim=2,onesided=True,normalized=False)[:,:,:,:-1]


