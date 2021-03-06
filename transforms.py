from torchvision import transforms
from PIL import ImageDraw,Image

class CircularMask(object):
  def __init__(self,size=256):
    self.size = size
    self.mask = Image.new("L", (size,size), 0)
    self.background = Image.new("RGB",(size,size),0)
    draw = ImageDraw.Draw(self.mask)
    draw.ellipse((0, 0, size, size), fill=255)
  
  def __call__(self,image):
    out = Image.composite(image,self.background,self.mask)
    return out

  def __str__(self):
    return f"CircularMask({self.size})"

#ImageNet color statistics
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
preprocess_fn = transforms.Compose(
      [transforms.ToTensor(),
       transforms.Normalize(mean, std)])