#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 20:24:40 2019

"""

# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains definitions for EfficientNet model.
[1] Mingxing Tan, Quoc V. Le
  EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
  ICML'19, https://arxiv.org/abs/1905.11946
"""


import collections
import math
import torch
import torch.nn as nn

GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate', 'data_format',
    'num_classes', 'width_coefficient', 'depth_coefficient', 'depth_divisor',
    'min_depth', 'drop_connect_rate', 'relu_fn', 'batch_norm', 'use_se',
    'local_pooling', 'condconv_num_experts'
])
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)

BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'strides', 'se_ratio', 'conv_type', 'fused_conv',
    'super_pixel', 'condconv'
])
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)

class Swish(nn.Module):
  """
  Funcion de activacion de: 
    Searching for Activation Functions
    https://arxiv.org/abs/1710.05941
  """
  def __init__(self):
    super(Swish, self).__init__()
    
  def forward(self,x):
    return x*torch.sigmoid(x)
  
  
class StochasticDepth(nn.Module):
  """
  Aleatoriamente desactiva la parte central de un 
  bloque de convoluciones como se describe en:
    Deep Networks with Stochastic Depth:
    https://arxiv.org/pdf/1603.09382.pdf
  """
  def __init__(self,drop_connect_rate):
    """
    p:probalidad de desactivar un bloque
    """    
    super(StochasticDepth,self).__init__()
    self.survival_prob = 1-drop_connect_rate 
    
  def forward(self,x):
    
    #A diferencia de "Stochastic Depth", la implementacion de EfficientNet
    #pasa la recalibracion de la inferencia al entrenamiento de modo similar
    #a dropout
    
    #Durante la evaluacion se mantienen todas las conexiones.
    if not self.training:
      return x
    
    #Las conexiones eliminadas son independientes en un mismo "minibatch"
    pop = torch.rand(x.shape[0],1,1,1,device=x.device)+self.survival_prob
    binary_tensor = torch.floor(pop)
    return x*binary_tensor/self.survival_prob
    
  def __repr__(self):
    return f"StochasticDepth({self.survival_prob})"
      

def round_filters(filters, global_params):
  """Round number of filters based on depth multiplier."""
  multiplier = global_params.width_coefficient
  divisor = global_params.depth_divisor
  min_depth = global_params.min_depth
  if not multiplier:
    return filters

  filters *= multiplier
  min_depth = min_depth or divisor
  new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
  # Make sure that round down does not go down by more than 10%.
  if new_filters < 0.9 * filters:
    new_filters += divisor
  return int(new_filters)

def round_repeats(repeats, global_params):
  """Round number of filters based on depth multiplier."""
  multiplier = global_params.depth_coefficient
  if not multiplier:
    return repeats
  return int(math.ceil(multiplier * repeats))

def Tensorflow_valid_padding(filter_height,filter_width,strides,in_height=None,in_width=None):
  """
  En los casos asimetricos se asigna un pixel adicional a la parte baja y la parte derecha de la imagen
  De:https://web.archive.org/web/20180502215629/https://www.tensorflow.org/api_guides/python/nn
  """
  if not in_height and not in_width:
    #En esta arquitectura las convoluciones espaciadas(strided) solo se usan en canales de dimension par.
    in_height = 2
    in_width  = 2
  
  if (in_height % strides[0] == 0):
    pad_along_height = max(filter_height - strides[0], 0)
  else:
    pad_along_height = max(filter_height - (in_height % strides[0]), 0)
  if (in_width % strides[1] == 0):
    pad_along_width = max(filter_width - strides[1], 0)
  else:
    pad_along_width = max(filter_width - (in_width % strides[1]), 0)
    
    
  pad_top = pad_along_height // 2
  pad_bottom = pad_along_height - pad_top
  pad_left = pad_along_width // 2
  pad_right = pad_along_width - pad_left
  
  return pad_left,pad_right,pad_top,pad_bottom


class MBConvBlock(nn.Module):
  """Constructor de un bloque de Mobile Inverted Residual Bottleneck"""
  def __init__(self,block_args,global_params,drop_connect_rate = None):
    """Initializes a MBConv block.
    Args:
      block_args: BlockArgs, argumentos especificos del bloque
      global_params: GlobalParams, comun a todos los bloques
    """
    
    super(MBConvBlock, self).__init__()
    self._block_args = block_args
    self._batch_norm_momentum = global_params.batch_norm_momentum
    self._batch_norm_epsilon = global_params.batch_norm_epsilon
    self._batch_norm = global_params.batch_norm
    self._relu_fn = global_params.relu_fn or Swish()
    self._has_se = (
        global_params.use_se and self._block_args.se_ratio is not None and
        0 < self._block_args.se_ratio <= 1)
    self.drop_connect_rate = drop_connect_rate
    self._build()
    
  def block_args(self):
    return self._block_args
    
  def _build(self):
    in_channels = self._block_args.input_filters
    expanded_channels = self._block_args.input_filters * self._block_args.expand_ratio
    kernel_size = self._block_args.kernel_size
    
    if self.drop_connect_rate:
      self._drop_connect = StochasticDepth(self.drop_connect_rate)
    
    if self._block_args.expand_ratio != 1:
      # Expansion phase.   
      self._expand_conv = nn.Conv2d(
        in_channels, 
        expanded_channels, 
        kernel_size = [1,1], 
        stride=1, 
        bias=False)
      self._bn0 = nn.BatchNorm2d(expanded_channels,
                     eps=self._batch_norm_epsilon,
                     momentum=self._batch_norm_momentum)
      
    # Depth-wise convolution phase
    #output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides[i])
    self._pad = nn.ZeroPad2d(Tensorflow_valid_padding(kernel_size,kernel_size,self._block_args.strides))
    self._depthwise_conv = nn.Conv2d(
        expanded_channels,
        expanded_channels,
        kernel_size,
        stride=self._block_args.strides,
        bias=False,
        groups= expanded_channels)
    self._bn1 = nn.BatchNorm2d(expanded_channels,
                   eps=self._batch_norm_epsilon,
                   momentum=self._batch_norm_momentum)    
        
    if self._has_se:
      num_reduced_filters = max(
          1, int(self._block_args.input_filters * self._block_args.se_ratio))#1/24
     
      # Squeeze and Excitation layer.Equivalentes a las capas fc
      self._se_reduce = nn.Conv2d(
          expanded_channels,
          num_reduced_filters,
          kernel_size=[1, 1],
          bias=True)#padding
      self._se_expand = nn.Conv2d(
          num_reduced_filters,
          expanded_channels,
          kernel_size=[1, 1],
          bias=True)#padding
      
    # Output phase.
    output_channels = self._block_args.output_filters
    self._project_conv = nn.Conv2d(
        expanded_channels,
        output_channels,
        kernel_size=[1, 1],
        bias=False)
    self._bn2 = nn.BatchNorm2d(output_channels,
                   eps=self._batch_norm_epsilon,
                   momentum=self._batch_norm_momentum)

  def _call_se(self, input_tensor):
    """Call Squeeze and Excitation layer.
    Args:
      input_tensor: Tensor, a single input tensor for Squeeze/Excitation layer.dims=(batch,channel,H,W)
    Returns:
      A output tensor, which should have the same shape as input.
    """
    se_tensor = torch.mean(input_tensor, dim=[2,3], keepdims=True)
    se_tensor = self._se_expand(self._relu_fn(self._se_reduce(se_tensor)))
    return torch.sigmoid(se_tensor) * input_tensor
  
  def forward(self, inputs):
    
    x = inputs
    
    same_dims = (all(s == 1 for s in self._block_args.strides) 
                and  
                self._block_args.input_filters==self._block_args.output_filters)
    
    depthwise_conv_fn = self._depthwise_conv
    project_conv_fn = self._project_conv
    
    if self._block_args.expand_ratio != 1:
      expand_conv_fn = self._expand_conv
      x = self._relu_fn(self._bn0(expand_conv_fn(x)))   
    
    x = self._relu_fn(self._bn1(depthwise_conv_fn(self._pad(x))))

    if self._has_se:
      x = self._call_se(x)
    
    x = self._bn2(project_conv_fn(x))
    
    if self._block_args.id_skip and same_dims: 
      if self.drop_connect_rate:
        x = self._drop_connect(x)  
      x = torch.add(x, inputs)

    return x


class Model(nn.Module):
  
  def __init__(self, blocks_args=None, global_params=None):
    """
    Args:
      blocks_args: A list of BlockArgs to construct block modules.
      global_params: GlobalParams, a set of global parameters.
    Raises:
      ValueError: when blocks_args is not specified as a list.
    """
    super(Model, self).__init__()
    if not isinstance(blocks_args, list):
      raise ValueError('blocks_args should be a list.')
      
    self._global_params = global_params
    self._blocks_args = blocks_args
    self._relu_fn = global_params.relu_fn or Swish()
    self._build()
    
  def _build(self):
    self._blocks = []
    batch_norm_momentum = self._global_params.batch_norm_momentum
    batch_norm_epsilon = self._global_params.batch_norm_epsilon
    
    # Stem part.
    out_channels = round_filters(32, self._global_params)
    self._pad = nn.ZeroPad2d(Tensorflow_valid_padding(3,3,(2,2)))
    self._conv_stem = nn.Conv2d(
        in_channels=3,
        out_channels=out_channels,
        kernel_size=[3, 3],
        stride=[2, 2],
        bias=False)
    self._bn0 = nn.BatchNorm2d(out_channels,
               eps=batch_norm_epsilon,
               momentum=batch_norm_momentum)
    
    # Transforma etapas
    blocks_args = []
    for block_args in self._blocks_args:
      assert block_args.num_repeat > 0
      
      # Update block input and output filters based on depth multiplier.
      input_filters = round_filters(block_args.input_filters,
                                    self._global_params)
      output_filters = round_filters(block_args.output_filters,
                                     self._global_params)
      
      block_args = block_args._replace(
      input_filters=input_filters,
      output_filters=output_filters,
      num_repeat=round_repeats(block_args.num_repeat, self._global_params))
      blocks_args.append(block_args)
    ##
    
    self._blocks_args = blocks_args
    total_blocks = sum([block_args.num_repeat for block_args in self._blocks_args])   
    
    #Construye bloques
    block_id=0
    for block_args in self._blocks_args:
      # The first block needs to take care of stride and filter size increase.
      self._blocks.append(MBConvBlock(block_args, self._global_params))
      block_id+=1
      
      if block_args.num_repeat > 1:  # rest of blocks with the same block_arg
        block_args = block_args._replace(
          input_filters=block_args.output_filters, strides=[1, 1])
        
      for _ in range(block_args.num_repeat - 1):
        block_drop_rate = self._global_params.drop_connect_rate * block_id / total_blocks
        self._blocks.append(MBConvBlock(block_args, self._global_params,
                                        drop_connect_rate = block_drop_rate))
        block_id+=1
        
    self._blocks = nn.ModuleList(self._blocks)    
    # Head part.
    out_channels=4*block_args.output_filters
    self._conv_head = nn.Conv2d(
        in_channels=block_args.output_filters,
        out_channels=out_channels,
        kernel_size=[1, 1],
        bias=False)
    self._bn1 = nn.BatchNorm2d(out_channels,
               eps=batch_norm_epsilon,
               momentum=batch_norm_momentum)
    
    if self._global_params.num_classes:
      self._fc = nn.Linear(out_channels,self._global_params.num_classes)
    else:
      self._fc = None
    
    if self._global_params.dropout_rate > 0:
      self._dropout = nn.Dropout(self._global_params.dropout_rate)
    else:
      self._dropout = None
      
  def forward(self,inputs):
    # Calls Stem layers
    outputs = self._relu_fn(
        self._bn0(self._conv_stem(self._pad(inputs))))
    
    # Calls blocks.
    for block in self._blocks:
      outputs = block(outputs)
    
    # Calls final layers and returns logits.      
    outputs = self._relu_fn(self._bn1(self._conv_head(outputs)))
    outputs = torch.mean(outputs,dim=[2,3],keepdim=False)
    if self._dropout:
      outputs = self._dropout(outputs)
    outputs = self._fc(outputs)
    return outputs
      
      


