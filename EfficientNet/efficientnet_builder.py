# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Model Builder for EfficientNet.
De: https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_builder.py
Adaptado para pytorch, solo permanecen 
las funciones relacionadas con la descripcion de la arquitectura,
no implementado drop_connect,superpixel,fused_conv,condconv,local_pooling"""


import re

from efficientnet_model import Model,BlockArgs,GlobalParams
  
MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]


def efficientnet_params(model_name):
  """Get efficientnet params based on model name."""
  params_dict = {
      # (width_coefficient, depth_coefficient, resolution, dropout_rate)
      'efficientnet-b0': (1.0, 1.0, 224, 0.2),
      'efficientnet-b1': (1.0, 1.1, 240, 0.2),
      'efficientnet-b2': (1.1, 1.2, 260, 0.3),
      'efficientnet-b3': (1.2, 1.4, 300, 0.3),
      'efficientnet-b4': (1.4, 1.8, 380, 0.4),
      'efficientnet-b5': (1.6, 2.2, 456, 0.4),
      'efficientnet-b6': (1.8, 2.6, 528, 0.5),
      'efficientnet-b7': (2.0, 3.1, 600, 0.5),
  }
  return params_dict[model_name]


class BlockDecoder(object):
  """Block Decoder for readability."""

  def _decode_block_string(self, block_string):
    """Gets a block through a string notation of arguments."""
    assert isinstance(block_string, str)
    ops = block_string.split('_')
    options = {}
    for op in ops:
      splits = re.split(r'(\d.*)', op)
      if len(splits) >= 2:
        key, value = splits[:2]
        options[key] = value

    if 's' not in options or len(options['s']) != 2:
      raise ValueError('Strides options should be a pair of integers.')

    return BlockArgs(
        kernel_size=int(options['k']),
        num_repeat=int(options['r']),
        input_filters=int(options['i']),
        output_filters=int(options['o']),
        expand_ratio=int(options['e']),
        id_skip=('noskip' not in block_string),
        se_ratio=float(options['se']) if 'se' in options else None,
        strides=[int(options['s'][0]),
                 int(options['s'][1])],
        conv_type=int(options['c']) if 'c' in options else 0,
        fused_conv=int(options['f']) if 'f' in options else 0,
        super_pixel=int(options['p']) if 'p' in options else 0,
        condconv=('cc' in block_string))

  def decode(self, string_list):
    """Decodes a list of string notations to specify blocks inside the network.
    Args:
      string_list: a list of strings, each string is a notation of block.
    Returns:
      A list of namedtuples to represent blocks arguments.
    """
    assert isinstance(string_list, list)
    blocks_args = []
    for block_string in string_list:
      blocks_args.append(self._decode_block_string(block_string))
    return blocks_args


def efficientnet(width_coefficient=None,
                 depth_coefficient=None,
                 dropout_rate=0.2,
                 drop_connect_rate=0.2,
                 code=None):
  
  """Creates a efficientnet model."""
  blocks_args = [
      'r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',
      'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',
      'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',
      'r1_k3_s11_e6_i192_o320_se0.25',
  ]
  if code:
    blocks_args = code
  
  global_params = GlobalParams(
      batch_norm_momentum=0.01,#TF->Torch: bn_momentum = 1-bn_momentum
      batch_norm_epsilon=1e-3,
      dropout_rate=dropout_rate,
      drop_connect_rate=drop_connect_rate,
      data_format='channels_last',
      num_classes=1000,
      width_coefficient=width_coefficient,
      depth_coefficient=depth_coefficient,
      depth_divisor=8,
      min_depth=None,
      relu_fn=None,#Funcion de pytorch
      batch_norm=None,#No habilitado
      use_se=True)
  decoder = BlockDecoder()
  return decoder.decode(blocks_args), global_params


def get_model_params(model_name, override_params,code=None):
  """Get the block args and global params for a given model."""
  if model_name.startswith('efficientnet'):
    width_coefficient, depth_coefficient, _, dropout_rate = (
        efficientnet_params(model_name))
    blocks_args, global_params = efficientnet(
        width_coefficient, depth_coefficient, dropout_rate,code=code)
  else:
    raise NotImplementedError('model name is not pre-defined: %s' % model_name)

  if override_params:
    # ValueError will be raised here if override_params has fields not included
    # in global_params.
    global_params = global_params._replace(**override_params)

  return blocks_args, global_params

def get_model(model_name,num_classes,code=None):
  "model_name:efficientnet-b{n},0-7"
  blocks,globalArgs=get_model_params(model_name,{"num_classes":num_classes},code=code)
  return Model(blocks,globalArgs)
