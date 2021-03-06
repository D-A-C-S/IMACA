import torch

def train_model(model,dataloader,optimizer,criterion,
               scheduler=None,device=torch.device("cpu"),
               log_data=False):
  #log_data:decide si recolecta informacion por cada minibatche, dado el caso se devuelve losses,accuracies
  
  running_loss = 0.0
  running_corrects = 0
  
  if log_data:
    losses = []
    accs = []
  
  for i,sample in enumerate(dataloader):

    optimizer.zero_grad()

    input_images = sample[0].to(device)
    target = sample[1].to(device)

    out = model(input_images)
    loss = criterion(out,target)

    loss.backward()

    if scheduler:
      scheduler.step()

    optimizer.step()

    _, preds = torch.max(out, 1)
    
    current_loss = loss.item()
    current_acc = torch.sum(preds == target.data)
    
    running_loss += current_loss * target.shape[0]
    running_corrects += current_acc
    
    if log_data:
      normalized_acc = current_acc.float()/target.shape[0]
      losses.append(current_loss)
      accs.append(normalized_acc.item())
    
    
  ##  
  epoch_loss = running_loss / len(dataloader.dataset)
  epoch_acc = running_corrects.item() / len(dataloader.dataset)
  print('train_loss: {:.3f} acc:{:.3f}'.format(epoch_loss,epoch_acc))
        
  if log_data:
    return losses,accs
  
  return epoch_loss,epoch_acc
  
@torch.no_grad()
def eval_model(model,dataloader,criterion,
               device=torch.device("cpu"),test=False):
  
  running_loss = 0.0
  running_corrects = 0
  best_loss = 1E5
  
  if test:
    all_labels = []
    all_predictions = []
    all_probas = []

  for i,sample in enumerate(dataloader):

    input_images = sample[0].to(device)
    target = sample[1].to(device)

    out = model(input_images)
    loss = criterion(out,target)
    
    probas = torch.nn.functional.softmax(out,dim=1)
    _, preds = torch.max(probas, 1)
    
    if test:
      all_labels+=list(target.detach().cpu().numpy().ravel())
      all_predictions+=list(preds.detach().cpu().numpy().ravel())
      all_probas.append(probas.detach().cpu().numpy())
      
    current_loss = loss.item()
    current_acc = torch.sum(preds == target.data)
    
    running_loss += current_loss * target.shape[0]
    running_corrects += current_acc
    
  ##endfor
  epoch_loss = running_loss / len(dataloader.dataset)
  epoch_acc = running_corrects.item() / len(dataloader.dataset)
  
  print('eval loss: {:.3f} acc:{:.3f}'.format(epoch_loss,epoch_acc))
  
  if (epoch_loss<best_loss) and not test:
    best_loss = epoch_loss
    torch.save({'state_dict':model.state_dict()},"Outputs/modelo.pt")
  
  if test:
    return all_labels, all_predictions,all_probas
  
  return epoch_loss,epoch_acc

def unlock_gradients(modelo,modulos,verbose=False):
  """Habilita el calculo de gradientes para los parametros 
  pertenecientes a los modulos de entrada"""

  for parameter in modelo.parameters():
    parameter.requires_grad = False
  
  for modulo in modulos:
    for parameter in modulo.parameters():
      parameter.requires_grad = True

  parametros_optimizables = []
  for name,parameter in modelo.named_parameters():
    if parameter.requires_grad:
      parametros_optimizables.append(parameter)
      if verbose:
        print(name)
  
  return parametros_optimizables
      
def freeze_base(modelo,modulos_libres):
  """Pone los modulos congelados en modo evaluacion.
  Y los modulos descongelados en modo entranamiento"""

  for name,modulo in modelo.named_modules():
    modulo.eval()
  
  for modulo in modulos_libres:
    modulo.train()



