import numpy as np
import torch
from .Losses import ECELoss

def ECE(all_probas,all_labels,all_predictions,n_bins=15):
  """
  Calcula el error medio de calibracion
  out:(0,1,2,3)
      0:Exactitud por segmento
      1:Probabilidad promedio por segmento
      2:Bandera indicando mediciones en el segmento
      3:error medio de calibracion

  """
  binary = np.array([label==pred for label,pred in zip(all_labels,all_predictions)])
  probabilidades = np.array(all_probas)
  bins = np.linspace(0,1.0,n_bins)
  #Posicion en el histograma para cada prediccion
  P=np.digitize(probabilidades,bins=bins)
  accs = []
  confs = []
  x = []
  ece = 0
  for i in range(n_bins):
    indices = np.asarray(P==i)
    if any(indices):
    #Para cada segmento del histograma,si existe una prediccion:
      probas = np.array(probabilidades[indices])
      binars = np.array(binary[indices])
      #Se calcula en promedio con cuanta seguridad se hacen las predicciones
      #y cuantos errores ocurren en el segmento
      accuracy = binars.mean()
      confidence = probas.mean()
      #Se normaliza de modo que cada prediccion tenga el mismo peso,
      #en el calculo del error esperado de calibracion.
      ece+= abs(accuracy-confidence)*len(probas)/len(probabilidades)

      accs.append(accuracy)
      confs.append(confidence)
      x.append(True)
    else:
      accs.append(0)
      confs.append(0)
      x.append(False)
      
  return accs,confs,x,ece

def ECEcheck(all_probas,all_labels,all_predictions,n_bins=15):
  ecetorch = ECELoss(n_bins)
  binary = np.array([label==pred for label,pred in zip(all_labels,all_predictions)])
  binary = torch.tensor(binary)
  all_probas = torch.tensor(all_probas)
  return ecetorch(all_probas,binary)
