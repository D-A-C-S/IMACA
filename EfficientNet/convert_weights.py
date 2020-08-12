#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:13:39 2019

@author: lejo
"""

import tensorflow as tf
import torch
import re
import argparse
import efficientnet_builder 


def ComprobacionNombres(pt_vars,tf_vars):
  count=0
  pt_names = list(zip(*pt_vars))[0]
  for name,ele in tf_vars:
    if name not in pt_names:
      count+=1
  return count

def ComprobacionNParametros(PT_vars,TF_vars):
  count=0
  for name,para in TF_vars.items():
    if PT_vars[name].numel()!=para.size:
      count+=1
  return count

def ComprobacionDimensiones(TF_vars,PT_vars):
  count=0
  for name in TF_vars.keys():
    if list(PT_vars[name].shape)!=list(TF_vars[name].shape):
      count+=1
    return count
  
def rp_name(lista,viejo,nuevo):
  return [(name.replace(viejo,nuevo),param) for name,param in lista]

def rp_block(cadena):
  return re.sub("_blocks_([0-9]?[0-9])\/",r'_blocks.\1.',cadena)



if __name__=="__main__":
  N_CLASSES = 1000
  parser = argparse.ArgumentParser()
  parser.add_argument("NumModelo",help="Numero del modelo,[0,7]")
  parser.add_argument("ModelWeights",help = "Filepath de los pesos del modelo con extension .ckpt")
  args = parser.parse_args()
  N_MODELO = args.NumModelo
  tf_path = args.ModelWeights
  
  #Construccion modelo de pytorch
  PT_modelo = efficientnet_builder.get_model(f"efficientnet-b{N_MODELO}",N_CLASSES)
  pt_vars = list(PT_modelo.state_dict().items()) 
  
  #Extraccion de pesos del modelo de Tensorflow
  tf_vars = []
  init_vars = tf.train.list_variables(tf_path)
  for name, shape in init_vars:
      array = tf.train.load_variable(tf_path, name)
      tf_vars.append((name, array.squeeze()))
      
  #Conversion de nombres al formato de pytorch
  tf_vars = [(name,param) for name,param in tf_vars if "ExponentialMovingAverage" not in name]
  init_vars = [(name,param) for name,param in init_vars if "ExponentialMovingAverage" not in name]
  tf_vars = rp_name(tf_vars,f"efficientnet-b{N_MODELO}/","_")
  tf_vars = [(rp_block(name),param) for name,param in tf_vars]
  tf_vars = rp_name(tf_vars,"depthwise_conv2d/depthwise_kernel","_depthwise_conv.weight")
  tf_vars = rp_name(tf_vars,"/",".")
  tf_vars = rp_name(tf_vars,".kernel",".weight")
  tf_vars = rp_name(tf_vars,"se.conv2d.","_se_reduce.")
  tf_vars = rp_name(tf_vars,"se.conv2d_1.","_se_expand.")
  
  #Conversion de nombres,condicional de los elementos del bloque
  Bloques = []
  bloques = [name for name,_ in PT_modelo._blocks.named_children()]
  for bloque in bloques:
    ElBloque =[idx for idx,_ in enumerate(tf_vars) if "_blocks."+str(bloque)+"." in tf_vars[idx][0]]
    if len(ElBloque):
      b=max(ElBloque)+1
      a=min(ElBloque)
      if not any([("conv2d_1.weight" in capa) for capa,_ in tf_vars[a:b]]):
            tf_vars[a:b] = rp_name(tf_vars[a:b],"conv2d.","_project_conv.")
            tf_vars[a:b] = rp_name(tf_vars[a:b],"tpu_batch_normalization.","_bn1.")
            tf_vars[a:b] = rp_name(tf_vars[a:b],"tpu_batch_normalization_1.","_bn2.")
      else:
        tf_vars[a:b] = rp_name(tf_vars[a:b],"conv2d.","_expand_conv.")
        tf_vars[a:b] = rp_name(tf_vars[a:b],"conv2d_1.","_project_conv.")
        tf_vars[a:b] = rp_name(tf_vars[a:b],"tpu_batch_normalization.","_bn0.")
        tf_vars[a:b] = rp_name(tf_vars[a:b],"tpu_batch_normalization_1.","_bn1.")
        tf_vars[a:b] = rp_name(tf_vars[a:b],"tpu_batch_normalization_2.","_bn2.")
    Bloques.append(ElBloque)
    
  #Conversion BatchNormalization y elementos fuera de los bloques  
  tf_vars = rp_name(tf_vars,"_head.tpu_batch_normalization.","_bn1.")
  tf_vars = rp_name(tf_vars,"_stem.tpu_batch_normalization.","_bn0.")
  tf_vars = rp_name(tf_vars,"_head.dense.","_fc.")
  tf_vars = rp_name(tf_vars,"_head.conv2d.","_conv_head.")
  tf_vars = rp_name(tf_vars,"_stem.conv2d.","_conv_stem.")
  tf_vars = rp_name(tf_vars,".moving_mean",".running_mean")
  tf_vars = rp_name(tf_vars,".moving_variance",".running_var")
  tf_vars = rp_name(tf_vars,".gamma",".weight")
  tf_vars = rp_name(tf_vars,".beta",".bias")
  
  assert ComprobacionNombres(pt_vars,tf_vars)==0
  #Diccionarios de variables y dimensiones
  tf_sizes = list(zip(*init_vars))[1]
  tf_names = list(zip(*tf_vars))[0]
  TF_vars  = {name:parameter for name,parameter in tf_vars}
  TF_sizes = {name:size for name,size in zip(tf_names,tf_sizes)}
  PT_vars  = PT_modelo.state_dict()
  
  assert ComprobacionNParametros(PT_vars,TF_vars)==0
  #Redimensionamiento de tensores 
  for name in TF_vars.keys():
    TF_vars[name] = TF_vars[name].reshape(TF_sizes[name]) 
    if list(PT_vars[name].shape)!=list(TF_vars[name].shape):
      
      if len(TF_vars[name].shape)==4:
        
        if "depthwise_" not in name:#Convolucion normal:TF(H,W,C_in,C_out);PT(C_out,C_in,H,W)
          TF_vars[name] = TF_vars[name].transpose(3,2,0,1)
          
        else:#Convolucion por grupos:TF(H,W,C_in=C_out,Mult);PT(C_in=C_out,Mult,H,W)
          TF_vars[name] = TF_vars[name].transpose(2,3,0,1)
          
      if len(TF_vars[name].shape)==2:#Capas FC
        TF_vars[name] = TF_vars[name].T
   
  assert ComprobacionDimensiones(TF_vars,PT_vars)==0     
  #Asignacion de pesos al diccionario de estados      
  for name in TF_vars.keys():
    PT_vars[name] = torch.from_numpy(TF_vars[name])
  
  print(PT_modelo.load_state_dict(PT_vars,strict=True))
  
  #Guardar pesos en formato pytorch
  torch.save({
            'model_state_dict': PT_modelo.state_dict(),
            }, tf_path[:-4]+"pth")
  
  
