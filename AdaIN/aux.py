from . import net
from .function import adaptive_instance_normalization
import torch
import torch.nn as nn

class Stylizer(nn.Module):
  "Adaptado de pytorch-AdaIN/test.py"
  def __init__(self,encoder_path,decoder_path):
    super().__init__()
    decoder = net.decoder
    vgg = net.vgg

    decoder.eval()
    vgg.eval()

    decoder.load_state_dict(torch.load(decoder_path))
    vgg.load_state_dict(torch.load(encoder_path))
    vgg = nn.Sequential(*list(vgg.children())[:31])

    self.encoder = vgg
    self.decoder = decoder

  def forward(self,content,style,alpha=1.0):
    """
    Opera sobre imagenes representadas por tensores 
    de dimensiones (B,C,H,W) en el rango de 0.0 a 1.0
    """

    content_f = self.encoder(content)
    style_f = self.encoder(style)      
    feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    out = self.decoder(feat)

    return out

