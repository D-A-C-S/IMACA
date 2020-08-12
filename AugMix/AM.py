from . import augmentations
from .aux import CircularMask
from torchvision import transforms
import torch.nn.functional as F
import torch
import numpy as np
from random import uniform
MIXTURE_WIDTH = 3
MIXTURE_DEPTH = -1
AUG_SEVERITY = 1#0-10

def aug(image, preprocess):
  """Perform AugMix augmentations and compute mixture.
  Args:
    image: PIL.Image input image
    preprocess: Preprocessing function which should return a torch tensor.
  Returns:
    mixed: Augmented and mixed image.
  """
  ws = np.float32(np.random.dirichlet([1] * MIXTURE_WIDTH))
  m = np.float32(np.random.beta(1, 1))

  mix = torch.zeros_like(preprocess(image))
  for i in range(MIXTURE_WIDTH):
    image_aug = image.copy()
    depth = MIXTURE_DEPTH if MIXTURE_DEPTH > 0 else np.random.randint(
        1, 4)
    for _ in range(depth):
      op = np.random.choice(augmentations.augmentations)
      image_aug = op(image_aug, AUG_SEVERITY)
    # Preprocessing commutes since all coefficients are convex
    mix += ws[i] * preprocess(image_aug)

  mixed = (1 - m) * preprocess(image) + m * mix
  return mixed

class AugMixDataset(torch.utils.data.Dataset):
  """Dataset wrapper to perform AugMix augmentation."""

  def __init__(self, dataset, preprocess, jsd=True,ratio=1.0):
    """
    ratio: probabilidad de aplicar la cadena de transformaciones
    """
    self.dataset = dataset
    self.preprocess = preprocess
    self.jsd = jsd
    self.ratio = ratio
  def __getitem__(self, i):
    sample = self.dataset[i]

    if not self.jsd and uniform(0,1)>self.ratio:
      sample["image"] = self.preprocess(sample["image"]) 
    elif not self.jsd:
      sample["image"] = aug(sample["image"], self.preprocess)
    else:
      image = self.preprocess(sample["image"])
      sample["aug1"]  = aug(sample["image"], self.preprocess)
      sample["aug2"]  = aug(sample["image"], self.preprocess)
      sample["image"] = image
                        
    return sample

  def __len__(self):
    return len(self.dataset)

def JSD(logits_clean,logits_aug1,logits_aug2):
  p_clean, p_aug1, p_aug2 = F.softmax(
      logits_clean, dim=1), F.softmax(
          logits_aug1, dim=1), F.softmax(
              logits_aug2, dim=1)
  p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
  loss = (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
          F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
          F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.
  return loss

def train(model,device,dataloader,optimizer,criterion,
          scheduler=None,jsd=False,batch_transforms=None,
          log_data=False):
  
  
  running_loss = 0.0
  running_corrects = 0
  if log_data:
    all_losses=[]
    all_accs = []
    
  for i,sample in enumerate(dataloader):

    optimizer.zero_grad()
    

    sample = {key:ele.to(device) for key,ele in sample.items()
              if isinstance(ele,torch.Tensor)}
    
    if batch_transforms:
      batch_transforms(sample)
    
    out = model(sample["image"])
    loss = criterion(out,sample["label"])

    if jsd:
      logit1 = model(sample["aug1"])
      logit2 = model(sample["aug2"])
      loss+=12*JSD(out,logit1,logit2)
    loss.backward()

    if scheduler:
      scheduler.step()

    optimizer.step()

    _, preds = torch.max(out, 1)
    
    current_loss = loss.item()
    current_acc = torch.sum(preds == sample["label"].data)

    if log_data:
      all_losses.append(current_loss)
      all_accs.append(current_acc.float()/sample["label"].shape[0])
    
    running_loss += current_loss * sample["label"].shape[0]
    running_corrects += current_acc
    
    
  ##  
  epoch_loss = running_loss / len(dataloader.dataset)
  epoch_acc = running_corrects.item() / len(dataloader.dataset)
  print('train_loss: {:.3f} acc:{:.3f}'.format(epoch_loss,epoch_acc))


  if log_data:
    return all_losses,all_accs




#ImageNet color statistics
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
BILINEAR = 2

def train_transforms(RCrop):
  return transforms.Compose(
      [transforms.RandomResizedCrop(RCrop,scale=(0.8, 1.0),ratio=(1.0,1.0)),
       transforms.RandomHorizontalFlip()])

preprocess = transforms.Compose(
      [transforms.ToTensor(),
       transforms.Normalize(mean, std)])

def test_transforms(RSize,RCrop):
  return transforms.Compose([
      transforms.Resize(RSize),
      transforms.CenterCrop(RCrop),
      preprocess])
