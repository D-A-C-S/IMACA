# coding: utf-8

import torch
import torch.nn as nn
import torchvision.transforms as T

from torchvision.datasets import  ImageFolder
from torch.utils.data import DataLoader
from transforms import preprocess_fn,CircularMask

from efficientnet_pytorch import EfficientNet
from optimizacion import unlock_gradients,freeze_base,train_model,eval_model


from sklearn.metrics import classification_report,confusion_matrix,matthews_corrcoef
from seaborn import heatmap

import yaml
#Loading configuration
with open('config.yaml') as f:
    C = yaml.safe_load(f)

batch_size   = C['batch_size']
epochs       = C['epochs']
lr           = C['learning_rate']
weight_decay = C['weight_decay']
weight_decay/=lr

#Data loading and transforms
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

train_transforms = T.Compose([T.RandomResizedCrop(C['crop_size'],scale=(0.8,1.0),ratio=(1.0,1.0)),
                              T.RandomHorizontalFlip(),
                              CircularMask(C['crop_size']),
                              T.RandomRotation(180,2),
                              preprocess_fn])


dev_transforms = T.Compose([T.Resize(int(1.35*C["image_size"])),
                            T.CenterCrop(C['crop_size']),
                            preprocess_fn])

train_dataset = ImageFolder(C['train_folder'],train_transforms)
dev_dataset   = ImageFolder(C['dev_folder'],dev_transforms)#

drop_last = len(train_dataset)%batch_size<batch_size/2

train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                          shuffle=True,drop_last=drop_last)
dev_loader   = DataLoader(dev_dataset, batch_size=batch_size, 
                        shuffle=False)
print(f"Especies:{train_dataset.classes}")


# Model definition


def build_model(num_classes):
    model = EfficientNet.from_pretrained('efficientnet-b1')
    model._fc = nn.Linear(model._fc.in_features,num_classes)#Inicializacion aleatoria
    model._dropout = nn.Dropout(C['dropout_prob'])
    return model.to(device)

#Freezing layers

model = build_model(len(train_dataset.classes))
modulos = ( [model._bn0,model._conv_stem]+
             list(model._blocks) +
            [model._bn1,model._conv_head,
            model._dropout,model._fc])

if C["num_frozen_layers"]>=len(modulos):
    raise ValueError(f"Todas las capas estan congeladas, max_num_frozen_layers={len(modulos)}")
modulos_libres = modulos[C["num_frozen_layers"]:]

      
parametros_optimizables = unlock_gradients(model,modulos_libres)
Nparametros = sum([para.numel() for name,para in model.named_parameters() if para.requires_grad])
print(f"Ajustando {Nparametros/1E6:.2f}M de parametros")


# Optimizacion


optimizer = torch.optim.AdamW(parametros_optimizables,weight_decay = weight_decay,lr=lr)
tao = epochs*(len(train_dataset)//batch_size + (not drop_last))
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,tao)

criterion = nn.CrossEntropyLoss()


for epoch in range(epochs):
  print('epoch {}/{}'.format(epoch+1,epochs))

  freeze_base(model,modulos_libres)
  train_model(model,train_loader,optimizer,
              criterion,scheduler=scheduler,device=device)
  
  
  model.eval()
  eval_loss,eval_acc = eval_model(model,dev_loader,criterion,
                                  device=device)


# Evaluacion


model.to(device)
model.eval()
all_labels, all_predictions,all_probas = eval_model(model,dev_loader,criterion,test=True,device=device)
print(classification_report(all_labels,all_predictions, target_names=dev_dataset.classes))
mcc = matthews_corrcoef(all_labels,all_predictions)
print(f"mcc:{mcc:.2f}")


mat = confusion_matrix(all_labels, all_predictions)
ax = heatmap(mat,
             xticklabels=dev_dataset.classes,
             yticklabels=dev_dataset.classes,
             annot=True,
             fmt='d')
ax.figure.savefig("Outputs/matriz_confusion.png")