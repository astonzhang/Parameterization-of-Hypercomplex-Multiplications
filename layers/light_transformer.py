# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
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

"""Commonly re-used transformer layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from ..layers import attention
# from tensor2tensor.layers import common_attention
from ..layers import light_attention as common_attention
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import expert_utils
from tensor2tensor.utils import mlperf_log
from ..layers.qlib import *

import tensorflow as tf

def transformer_prepare_encoder(inputs, target_space, hparams, features=None):
  """Prepare one shard of the model for the encoder.

  Args:
    inputs: a Tensor.
    target_space: a Tensor.
    hparams: run hyperparameters
    features: optionally pass the entire features dictionary as well.
      This is needed now for "packed" datasets.

  Returns:
    encoder_input: a Tensor, bottom of encoder stack
    encoder_self_attention_bias: a bias tensor for use in encoder self-attention
    encoder_decoder_attention_bias: a bias tensor for use in encoder-decoder
      attention
  """
  ishape_static = inputs.shape.as_list()
  encoder_input = inputs
  if features and "inputs_segmentation" in features:
    # Packed dataset.  Keep the examples from seeing each other.
    inputs_segmentation = features["inputs_segmentation"]
    inputs_position = features["inputs_position"]
    targets_segmentation = features["targets_segmentation"]
    encoder_self_attention_bias = common_attention.attention_bias_same_segment(
        inputs_segmentation, inputs_segmentation)
    encoder_decoder_attention_bias = (
        common_attention.attention_bias_same_segment(targets_segmentation,
                                                     inputs_segmentation))
  else:
    # Usual case - not a packed dataset.
    encoder_padding = common_attention.embedding_to_padding(encoder_input)
    ignore_padding = common_attention.attention_bias_ignore_padding(
        encoder_padding)
    encoder_self_attention_bias = ignore_padding
    encoder_decoder_attention_bias = ignore_padding
    inputs_position = None
  if hparams.proximity_bias:
    encoder_self_attention_bias += common_attention.attention_bias_proximal(
        common_layers.shape_list(inputs)[1])
  if hparams.get("use_target_space_embedding", True):
    # Append target_space_id embedding to inputs.
    emb_target_space = common_layers.embedding(
        target_space,
        32,
        ishape_static[-1],
        name="target_space_embedding",
        dtype=tf.bfloat16
        if hparams.activation_dtype == "bfloat16" else tf.float32)
    emb_target_space = tf.reshape(emb_target_space, [1, 1, -1])
    encoder_input += emb_target_space
  if hparams.pos == "timing":
    if inputs_position is not None:
      encoder_input = common_attention.add_timing_signal_1d_given_position(
          encoder_input, inputs_position)
    else:
      encoder_input = common_attention.add_timing_signal_1d(encoder_input)
  elif hparams.pos == "emb":
    encoder_input = common_attention.add_positional_embedding(
        encoder_input, hparams.max_length, "inputs_positional_embedding",
        inputs_position)
  if hparams.activation_dtype == "bfloat16":
    encoder_self_attention_bias = tf.cast(encoder_self_attention_bias,
                                          tf.bfloat16)
    encoder_decoder_attention_bias = tf.cast(encoder_decoder_attention_bias,
                                             tf.bfloat16)
  return (encoder_input, encoder_self_attention_bias,
          encoder_decoder_attention_bias)


def transformer_encoder(encoder_input,
                        encoder_self_attention_bias,
                        hparams,
                        name="encoder",
                        nonpadding=None,
                        save_weights_to=None,
                        make_image_summary=True,
                        losses=None,
                        attn_bias_for_padding=None):
  """A stack of transformer layers.

  Args:
    encoder_input: a Tensor
    encoder_self_attention_bias: bias Tensor for self-attention
       (see common_attention.attention_bias())
    hparams: hyperparameters for model
    name: a string
    nonpadding: optional Tensor with shape [batch_size, encoder_length]
      indicating what positions are not padding.  This must either be
      passed in, which we do for "packed" datasets, or inferred from
      encoder_self_attention_bias.  The knowledge about padding is used
      for pad_remover(efficiency) and to mask out padding in convolutional
      layers.
    save_weights_to: an optional dictionary to capture attention weights
      for visualization; the weights tensor will be appended there under
      a string key created from the variable scope (including name).
    make_image_summary: Whether to make an attention image summary.
    losses: optional list onto which to append extra training losses
    attn_bias_for_padding: Padded attention bias in case a unidirectional
      encoder is being used where future attention is masked.

  Returns:
    y: a Tensors
  """
  x = encoder_input
  attention_dropout_broadcast_dims = (
      common_layers.comma_separated_string_to_integer_list(
          getattr(hparams, "attention_dropout_broadcast_dims", "")))
  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_NUM_HIDDEN_LAYERS,
      value=hparams.num_encoder_layers or hparams.num_hidden_layers)
  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_ATTENTION_DROPOUT,
      value=hparams.attention_dropout)
  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_ATTENTION_DENSE,
      value={
          "use_bias": "false",
          "num_heads": hparams.num_heads,
          "hidden_size": hparams.hidden_size
      })

  with tf.variable_scope(name):
    if nonpadding is not None:
      padding = 1.0 - nonpadding
    else:
      attention_bias = encoder_self_attention_bias
      if attn_bias_for_padding is not None:
        attention_bias = attn_bias_for_padding
      padding = common_attention.attention_bias_to_padding(attention_bias)
      nonpadding = 1.0 - padding
    pad_remover = None
    if hparams.use_pad_remover and not common_layers.is_xla_compiled():
      pad_remover = expert_utils.PadRemover(padding)
    res = None
    for layer in range(hparams.num_encoder_layers or hparams.num_hidden_layers):
      with tf.variable_scope("layer_%d" % layer):
        with tf.variable_scope("self_attention"):
            is_training = hparams.mode == tf.contrib.learn.ModeKeys.TRAIN
            y, res = common_attention.multihead_attention(
                  common_layers.layer_preprocess(x, hparams),
                  None,
                  encoder_self_attention_bias,
                  hparams.attention_key_channels or hparams.hidden_size,
                  hparams.attention_value_channels or hparams.hidden_size,
                  hparams.hidden_size,
                  hparams.num_heads,
                  hparams.attention_dropout,
                  attention_type=hparams.self_attention_type,
                  max_relative_position=hparams.max_relative_position,
                  heads_share_relative_embedding=(
                      hparams.heads_share_relative_embedding),
                  add_relative_to_values=hparams.add_relative_to_values,
                  save_weights_to=save_weights_to,
                  make_image_summary=make_image_summary,
                  dropout_broadcast_dims=attention_dropout_broadcast_dims,
                  max_length=hparams.get("max_length"),
                  vars_3d=hparams.get("attention_variables_3d"),
                  is_training=is_training,
                  light_mode=hparams.light_mode,
                  hparams=hparams,
                  res=res)
            x = common_layers.layer_postprocess(x, y, hparams)
        with tf.variable_scope("ffn"):
            y = transformer_ffn_layer(
                  common_layers.layer_preprocess(x, hparams),
                  hparams,
                  pad_remover,
                  conv_padding="SAME",
                  nonpadding_mask=nonpadding,
                  losses=losses)
            x = common_layers.layer_postprocess(x, y, hparams)

    # if normalization is done in layer_preprocess, then it should also be done
    # on the output, since the output can grow very large, being the sum of
    # a whole stack of unnormalized layer outputs.
    mlperf_log.transformer_print(
        key=mlperf_log.MODEL_HP_NORM,
        value={"hidden_size": hparams.hidden_size})
    return common_layers.layer_preprocess(x, hparams)

def factor_dense_relu_dense(inputs,
                     filter_size,
                     output_size,
                     output_activation=None,
                     dropout=0.0,
                     dropout_broadcast_dims=None,
                     layer_collection=None,
                     name=None):
  """Hidden layer with RELU activation followed by linear projection."""
  # layer_name is appended with "conv1" or "conv2" in this method only for
  # historical reasons. These are in fact dense layers.
  print("Using F-Dense")
  layer_name = "%s_{}" % name if name else "{}"
  h = factorized_ffn_3d(inputs, filter_size,
            name=layer_name.format('qconv1'), activation=tf.nn.relu)
  if dropout != 0.0:
    h = common_layers.dropout_with_broadcast_dims(
        h, 1.0 - dropout, broadcast_dims=dropout_broadcast_dims)
  o = factorized_ffn_3d(h, output_size,
             name=layer_name.format('qconv2'),
            activation=None)
  return o

def ftm_dense_relu_dense(inputs,
                     filter_size,
                     output_size,
                     output_activation=None,
                     dropout=0.0,
                     dropout_broadcast_dims=None,
                     layer_collection=None,
                     factor=4,
                     name=None):
  """Hidden layer with RELU activation followed by linear projection."""
  # layer_name is appended with "conv1" or "conv2" in this method only for
  # historical reasons. These are in fact dense layers.
  print("Using FTM-Dense")
  layer_name = "%s_{}" % name if name else "{}"
  # h = quarternion_ffn_3d(inputs, filter_size,
  #           name=layer_name.format('qconv1'), activation=tf.nn.relu)
  h, _ = common_attention.build_tensor_machines(inputs,
                    num_outputs=filter_size, k=factor,
                    name=layer_name.format('qconv1'), reshape=True)
  h = tf.nn.relu(h)
  if dropout != 0.0:
    h = common_layers.dropout_with_broadcast_dims(
        h, 1.0 - dropout, broadcast_dims=dropout_broadcast_dims)
  o = common_layers.dense(
      h,
      output_size,
      activation=output_activation,
      use_bias=True,
      # layer_collection=layer_collection,
      name=layer_name.format("conv2"))
  return o

def quaternion_dense_relu_dense(inputs,
                     filter_size,
                     output_size,
                     output_activation=None,
                     dropout=0.0,
                     dropout_broadcast_dims=None,
                     layer_collection=None,
                     name=None):
  """Hidden layer with RELU activation followed by linear projection."""
  # layer_name is appended with "conv1" or "conv2" in this method only for
  # historical reasons. These are in fact dense layers.
  print("Using Q-Dense")
  layer_name = "%s_{}" % name if name else "{}"
  h = quarternion_ffn_3d(inputs, filter_size,
            name=layer_name.format('qconv1'), activation=tf.nn.relu)
  if dropout != 0.0:
    h = common_layers.dropout_with_broadcast_dims(
        h, 1.0 - dropout, broadcast_dims=dropout_broadcast_dims)
  o = quarternion_ffn_3d(h, output_size,
             name=layer_name.format('qconv2'),
            activation=None)
  return o

def random_dense_relu_dense(inputs,
                     filter_size,
                     output_size,
                     output_activation=None,
                     dropout=0.0,
                     dropout_broadcast_dims=None,
                     layer_collection=None,
                     factor=4,
                     upsize=1,
                     dual=0,
                     name=None):
  """Hidden layer with RELU activation followed by linear projection."""
  # layer_name is appended with "conv1" or "conv2" in this method only for
  # historical reasons. These are in fact dense layers.
  print("Using R-Dense")
  layer_name = "%s_{}" % name if name else "{}"
  filter_size *= upsize
  h = random_ffn_3d(inputs, filter_size, n=factor,
            name=layer_name.format('rconv1'), activation=tf.nn.relu, dual=dual)
  if dropout != 0.0:
    h = common_layers.dropout_with_broadcast_dims(
        h, 1.0 - dropout, broadcast_dims=dropout_broadcast_dims)
  o = random_ffn_3d(h, output_size, n=factor,
             name=layer_name.format('rconv2'),
            activation=None, dual=dual)
  return o

def transformer_ffn_layer(x,
                          hparams,
                          pad_remover=None,
                          conv_padding="LEFT",
                          nonpadding_mask=None,
                          losses=None,
                          cache=None,
                          decode_loop_step=None,
                          readout_filter_size=0):
  """Feed-forward layer in the transformer.

  Args:
    x: a Tensor of shape [batch_size, length, hparams.hidden_size]
    hparams: hyperparameters for model
    pad_remover: an expert_utils.PadRemover object tracking the padding
      positions. If provided, when using convolutional settings, the padding
      is removed before applying the convolution, and restored afterward. This
      can give a significant speedup.
    conv_padding: a string - either "LEFT" or "SAME".
    nonpadding_mask: an optional Tensor with shape [batch_size, length].
      needed for convolutional layers with "SAME" padding.
      Contains 1.0 in positions corresponding to nonpadding.
    losses: optional list onto which to append extra training losses
    cache: dict, containing tensors which are the results of previous
        attentions, used for fast decoding.
    decode_loop_step: An integer, step number of the decoding loop.
        Only used for inference on TPU.
    readout_filter_size: if it's greater than 0, then it will be used instead of
      filter_size


  Returns:
    a Tensor of shape [batch_size, length, hparams.hidden_size]

  Raises:
    ValueError: If losses arg is None, but layer generates extra losses.
  """
  ffn_layer = hparams.ffn_layer
  relu_dropout_broadcast_dims = (
      common_layers.comma_separated_string_to_integer_list(
          getattr(hparams, "relu_dropout_broadcast_dims", "")))
  if ffn_layer == "conv_hidden_relu":
    # Backwards compatibility
    ffn_layer = "dense_relu_dense"
  if ffn_layer == "quaternion_dense_relu_dense":
    # In simple convolution mode, use `pad_remover` to speed up processing.
    mlperf_log.transformer_print(
        key=mlperf_log.MODEL_HP_FFN_FILTER_DENSE,
        value={
            "filter_size": hparams.filter_size,
            "use_bias": "True",
            "activation": mlperf_log.RELU
        })
    mlperf_log.transformer_print(
        key=mlperf_log.MODEL_HP_FFN_OUTPUT_DENSE,
        value={
            "hidden_size": hparams.hidden_size,
            "use_bias": "True",
        })
    mlperf_log.transformer_print(
        key=mlperf_log.MODEL_HP_RELU_DROPOUT, value=hparams.relu_dropout)
    if pad_remover:
      original_shape = common_layers.shape_list(x)
      # Collapse `x` across examples, and remove padding positions.
      x = tf.reshape(x, tf.concat([[-1], original_shape[2:]], axis=0))
      x = tf.expand_dims(pad_remover.remove(x), axis=0)
    conv_output = quaternion_dense_relu_dense(
        x,
        hparams.filter_size,
        hparams.hidden_size,
        dropout=hparams.relu_dropout,
        dropout_broadcast_dims=relu_dropout_broadcast_dims)
    if pad_remover:
      # Restore `conv_output` to the original shape of `x`, including padding.
      conv_output = tf.reshape(
          pad_remover.restore(tf.squeeze(conv_output, axis=0)), original_shape)
    return conv_output
  if ffn_layer == "random_dense_relu_dense":
    # In simple convolution mode, use `pad_remover` to speed up processing.
    mlperf_log.transformer_print(
        key=mlperf_log.MODEL_HP_FFN_FILTER_DENSE,
        value={
            "filter_size": hparams.filter_size,
            "use_bias": "True",
            "activation": mlperf_log.RELU
        })
    mlperf_log.transformer_print(
        key=mlperf_log.MODEL_HP_FFN_OUTPUT_DENSE,
        value={
            "hidden_size": hparams.hidden_size,
            "use_bias": "True",
        })
    mlperf_log.transformer_print(
        key=mlperf_log.MODEL_HP_RELU_DROPOUT, value=hparams.relu_dropout)
    if pad_remover:
      original_shape = common_layers.shape_list(x)
      # Collapse `x` across examples, and remove padding positions.
      x = tf.reshape(x, tf.concat([[-1], original_shape[2:]], axis=0))
      x = tf.expand_dims(pad_remover.remove(x), axis=0)
    conv_output = random_dense_relu_dense(
        x,
        hparams.filter_size,
        hparams.hidden_size,
        dropout=hparams.relu_dropout,
        factor=hparams.factor,
        upsize=hparams.ffn_upsize,
        dual=hparams.dual,
        dropout_broadcast_dims=relu_dropout_broadcast_dims)
    if pad_remover:
      # Restore `conv_output` to the original shape of `x`, including padding.
      conv_output = tf.reshape(
          pad_remover.restore(tf.squeeze(conv_output, axis=0)), original_shape)
    return conv_output

  if ffn_layer == 'factor_dense_relu_dense':
    # In simple convolution mode, use `pad_remover` to speed up processing.
    mlperf_log.transformer_print(
        key=mlperf_log.MODEL_HP_FFN_FILTER_DENSE,
        value={
            "filter_size": hparams.filter_size,
            "use_bias": "True",
            "activation": mlperf_log.RELU
        })
    mlperf_log.transformer_print(
        key=mlperf_log.MODEL_HP_FFN_OUTPUT_DENSE,
        value={
            "hidden_size": hparams.hidden_size,
            "use_bias": "True",
        })
    mlperf_log.transformer_print(
        key=mlperf_log.MODEL_HP_RELU_DROPOUT, value=hparams.relu_dropout)
    if pad_remover:
      original_shape = common_layers.shape_list(x)
      # Collapse `x` across examples, and remove padding positions.
      x = tf.reshape(x, tf.concat([[-1], original_shape[2:]], axis=0))
      x = tf.expand_dims(pad_remover.remove(x), axis=0)
    conv_output = factor_dense_relu_dense(
        x,
        hparams.filter_size,
        hparams.hidden_size,
        dropout=hparams.relu_dropout,
        dropout_broadcast_dims=relu_dropout_broadcast_dims)
    if pad_remover:
      # Restore `conv_output` to the original shape of `x`, including padding.
      conv_output = tf.reshape(
          pad_remover.restore(tf.squeeze(conv_output, axis=0)), original_shape)
    return conv_output
  elif ffn_layer == 'ftm_dense':
    mlperf_log.transformer_print(
        key=mlperf_log.MODEL_HP_FFN_FILTER_DENSE,
        value={
            "filter_size": hparams.filter_size,
            "use_bias": "True",
            "activation": mlperf_log.RELU
        })
    mlperf_log.transformer_print(
        key=mlperf_log.MODEL_HP_FFN_OUTPUT_DENSE,
        value={
            "hidden_size": hparams.hidden_size,
            "use_bias": "True",
        })
    mlperf_log.transformer_print(
        key=mlperf_log.MODEL_HP_RELU_DROPOUT, value=hparams.relu_dropout)
    if pad_remover:
      original_shape = common_layers.shape_list(x)
      # Collapse `x` across examples, and remove padding positions.
      x = tf.reshape(x, tf.concat([[-1], original_shape[2:]], axis=0))
      x = tf.expand_dims(pad_remover.remove(x), axis=0)
    conv_output = ftm_dense_relu_dense(
        x,
        hparams.filter_size,
        hparams.hidden_size,
        factor=hparams.factor,
        dropout=hparams.relu_dropout,
        dropout_broadcast_dims=relu_dropout_broadcast_dims)
    if pad_remover:
      # Restore `conv_output` to the original shape of `x`, including padding.
      conv_output = tf.reshape(
          pad_remover.restore(tf.squeeze(conv_output, axis=0)), original_shape)
    return conv_output

  elif ffn_layer == "dense_relu_dense":
    # In simple convolution mode, use `pad_remover` to speed up processing.
    mlperf_log.transformer_print(
        key=mlperf_log.MODEL_HP_FFN_FILTER_DENSE,
        value={
            "filter_size": hparams.filter_size,
            "use_bias": "True",
            "activation": mlperf_log.RELU
        })
    mlperf_log.transformer_print(
        key=mlperf_log.MODEL_HP_FFN_OUTPUT_DENSE,
        value={
            "hidden_size": hparams.hidden_size,
            "use_bias": "True",
        })
    mlperf_log.transformer_print(
        key=mlperf_log.MODEL_HP_RELU_DROPOUT, value=hparams.relu_dropout)
    if pad_remover:
      original_shape = common_layers.shape_list(x)
      # Collapse `x` across examples, and remove padding positions.
      x = tf.reshape(x, tf.concat([[-1], original_shape[2:]], axis=0))
      x = tf.expand_dims(pad_remover.remove(x), axis=0)
    conv_output = common_layers.dense_relu_dense(
        x,
        hparams.filter_size,
        hparams.hidden_size,
        dropout=hparams.relu_dropout,
        dropout_broadcast_dims=relu_dropout_broadcast_dims)
    if pad_remover:
      # Restore `conv_output` to the original shape of `x`, including padding.
      conv_output = tf.reshape(
          pad_remover.restore(tf.squeeze(conv_output, axis=0)), original_shape)
    return conv_output
  elif ffn_layer == "conv_relu_conv":
    return common_layers.conv_relu_conv(
        x,
        readout_filter_size or hparams.filter_size,
        hparams.hidden_size,
        first_kernel_size=hparams.conv_first_kernel,
        second_kernel_size=1,
        padding=conv_padding,
        nonpadding_mask=nonpadding_mask,
        dropout=hparams.relu_dropout,
        cache=cache,
        decode_loop_step=decode_loop_step)
  elif ffn_layer == "parameter_attention":
    return common_attention.parameter_attention(
        x, hparams.parameter_attention_key_channels or hparams.hidden_size,
        hparams.parameter_attention_value_channels or hparams.hidden_size,
        hparams.hidden_size, readout_filter_size or hparams.filter_size,
        hparams.num_heads,
        hparams.attention_dropout)
  elif ffn_layer == "conv_hidden_relu_with_sepconv":
    return common_layers.conv_hidden_relu(
        x,
        readout_filter_size or hparams.filter_size,
        hparams.hidden_size,
        kernel_size=(3, 1),
        second_kernel_size=(31, 1),
        padding="LEFT",
        dropout=hparams.relu_dropout)
  elif ffn_layer == "sru":
    return common_layers.sru(x)
  elif ffn_layer == "local_moe_tpu":
    overhead = (
        hparams.moe_overhead_train
        if hparams.mode == tf.estimator.ModeKeys.TRAIN else
        hparams.moe_overhead_eval)
    ret, loss = expert_utils.local_moe_tpu(
        x,
        hparams.filter_size // 2,
        hparams.hidden_size,
        hparams.moe_num_experts,
        overhead=overhead,
        loss_coef=hparams.moe_loss_coef)
  elif ffn_layer == "local_moe":
    overhead = (
        hparams.moe_overhead_train
        if hparams.mode == tf.estimator.ModeKeys.TRAIN else
        hparams.moe_overhead_eval)
    ret, loss = expert_utils.local_moe(
        x,
        True,
        expert_utils.ffn_expert_fn(hparams.hidden_size, [hparams.filter_size],
                                   hparams.hidden_size),
        hparams.moe_num_experts,
        k=hparams.moe_k,
        hparams=hparams)
    losses.append(loss)
    return ret
  else:
    assert ffn_layer == "none"
    return x
