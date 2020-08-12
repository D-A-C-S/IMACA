"""
Funciones sobre tensores aplicadas al objetivo
de optimizacion.
"""
def total_variation(image,Beta=1.0,eps=1E-8):
  "image:tensor of shape 1,3,?,?"
  x_grad = image.roll(1,dims=2)-image
  y_grad = image.roll(1,dims=3)-image
  TVR = (x_grad**2+y_grad**2+eps)**(Beta/2)
  return TVR.mean()
