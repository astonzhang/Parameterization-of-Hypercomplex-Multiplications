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

"""Transformer model from "Attention Is All You Need".

The Transformer model consists of an encoder and a decoder. Both are stacks
of self-attention layers followed by feed-forward layers. This model yields
good results on a number of problems, especially in NLP and machine translation.

See "Attention Is All You Need" (https://arxiv.org/abs/1706.03762) for the full
description of the model and the results obtained with its early version.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import range  # pylint: disable=redefined-builtin

from tensor2tensor.data_generators import librispeech
# from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers

from ..layers import light_attention as common_attention
from ..layers import light_transformer as transformer_layers
from tensor2tensor.utils import beam_search
from tensor2tensor.utils import expert_utils
from tensor2tensor.utils import mlperf_log
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf

from tensorflow.python.ops import inplace_ops
from tensorflow.python.util import nest


# Alias some commonly reused layers, here and elsewhere.
transformer_prepare_encoder = transformer_layers.transformer_prepare_encoder
transformer_encoder = transformer_layers.transformer_encoder
transformer_ffn_layer = transformer_layers.transformer_ffn_layer


@registry.register_model
class LightTransformer(t2t_model.T2TModel):
  """Attention net.  See file docstring."""

  def __init__(self, *args, **kwargs):
    super(LightTransformer, self).__init__(*args, **kwargs)
    self.attention_weights = dict()  # For visualizing attention heads.

  def encode(self, inputs, target_space, hparams, features=None, losses=None):
    """Encode transformer inputs.

    Args:
      inputs: Transformer inputs [batch_size, input_length, 1, hidden_dim] which
        will be flattened along the two spatial dimensions.
      target_space: scalar, target space ID.
      hparams: hyperparameters for model.
      features: optionally pass the entire features dictionary as well.
        This is needed now for "packed" datasets.
      losses: optional list onto which to append extra training losses

    Returns:
      Tuple of:
          encoder_output: Encoder representation.
              [batch_size, input_length, hidden_dim]
          encoder_decoder_attention_bias: Bias and mask weights for
              encoder-decoder attention. [batch_size, input_length]
    """
    if(hparams.cut_length==1):
        print("Cutting length..")
        inputs = inputs[:,:hparams.max_length, :, : ]
    inputs = common_layers.flatten4d3d(inputs)


    encoder_input, self_attention_bias, encoder_decoder_attention_bias = (
        transformer_prepare_encoder(
            inputs, target_space, hparams, features=features))

    mlperf_log.transformer_print(
        key=mlperf_log.MODEL_HP_LAYER_POSTPROCESS_DROPOUT,
        value=hparams.layer_prepostprocess_dropout)

    encoder_input = tf.nn.dropout(encoder_input,
                                  1.0 - hparams.layer_prepostprocess_dropout)

    encoder_output = transformer_encoder(
        encoder_input,
        self_attention_bias,
        hparams,
        nonpadding=features_to_nonpadding(features, "inputs"),
        save_weights_to=self.attention_weights,
        make_image_summary=not common_layers.is_xla_compiled(),
        losses=losses)

    return encoder_output, encoder_decoder_attention_bias

  def decode(self,
             decoder_input,
             encoder_output,
             encoder_decoder_attention_bias,
             decoder_self_attention_bias,
             hparams,
             cache=None,
             decode_loop_step=None,
             nonpadding=None,
             losses=None):
    """Decode Transformer outputs from encoder representation.

    Args:
      decoder_input: inputs to bottom of the model.
          [batch_size, decoder_length, hidden_dim]
      encoder_output: Encoder representation.
          [batch_size, input_length, hidden_dim]
      encoder_decoder_attention_bias: Bias and mask weights for
          encoder-decoder attention. [batch_size, input_length]
      decoder_self_attention_bias: Bias and mask weights for decoder
          self-attention. [batch_size, decoder_length]
      hparams: hyperparameters for model.
      cache: dict, containing tensors which are the results of previous
          attentions, used for fast decoding.
      decode_loop_step: An integer, step number of the decoding loop.
          Only used for inference on TPU.
      nonpadding: optional Tensor with shape [batch_size, decoder_length]
      losses: optional list onto which to append extra training losses

    Returns:
      Final decoder representation. [batch_size, decoder_length, hidden_dim]
    """
    mlperf_log.transformer_print(
        key=mlperf_log.MODEL_HP_LAYER_POSTPROCESS_DROPOUT,
        value=hparams.layer_prepostprocess_dropout)
    decoder_input = tf.nn.dropout(decoder_input,
                                  1.0 - hparams.layer_prepostprocess_dropout)

    decoder_output = transformer_decoder(
        decoder_input,
        encoder_output,
        decoder_self_attention_bias,
        encoder_decoder_attention_bias,
        hparams,
        cache=cache,
        decode_loop_step=decode_loop_step,
        nonpadding=nonpadding,
        save_weights_to=self.attention_weights,
        losses=losses)

    if (common_layers.is_xla_compiled() and
        hparams.mode == tf.estimator.ModeKeys.TRAIN):
      # TPU does not react kindly to extra dimensions.
      # TODO(noam): remove this once TPU is more forgiving of extra dims.
      return decoder_output
    else:
      # Expand since t2t expects 4d tensors.
      return tf.expand_dims(decoder_output, axis=2)

  def body(self, features):
    """Transformer main model_fn.

    Args:
      features: Map of features to the model. Should contain the following:
          "inputs": Transformer inputs.
              [batch_size, input_length, 1, hidden_dim].
          "targets": Target decoder outputs.
              [batch_size, decoder_length, 1, hidden_dim]
          "target_space_id": A scalar int from data_generators.problem.SpaceID.

    Returns:
      Final decoder representation. [batch_size, decoder_length, hidden_dim]
    """
    hparams = self._hparams

    losses = []

    if self.has_input:
      inputs = features["inputs"]
      target_space = features["target_space_id"]
      encoder_output, encoder_decoder_attention_bias = self.encode(
          inputs, target_space, hparams, features=features, losses=losses)
    else:
      encoder_output, encoder_decoder_attention_bias = (None, None)

    targets = features["targets"]
    targets_shape = common_layers.shape_list(targets)
    targets = common_layers.flatten4d3d(targets)
    decoder_input, decoder_self_attention_bias = transformer_prepare_decoder(
        targets, hparams, features=features)
    decoder_output = self.decode(
        decoder_input,
        encoder_output,
        encoder_decoder_attention_bias,
        decoder_self_attention_bias,
        hparams,
        nonpadding=features_to_nonpadding(features, "targets"),
        losses=losses)

    expected_attentions = features.get("expected_attentions")
    if expected_attentions is not None:
      attention_loss = common_attention.encoder_decoder_attention_loss(
          expected_attentions, self.attention_weights,
          hparams.expected_attention_loss_type,
          hparams.expected_attention_loss_multiplier)
      return decoder_output, {"attention_loss": attention_loss}

    ret = tf.reshape(decoder_output, targets_shape)
    if losses:
      return ret, {"extra_loss": tf.add_n(losses)}
    else:
      return ret

  def _greedy_infer(self, features, decode_length, use_tpu=False):
    """Fast version of greedy decoding.

    Args:
      features: an map of string to `Tensor`
      decode_length: an integer.  How many additional timesteps to decode.
      use_tpu: A bool. Whether to build the inference graph for TPU.

    Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, <= decode_length] if beam_size == 1 or
              [batch_size, top_beams, <= decode_length]
          "scores": decoding log probs from the beam search,
              None if using greedy decoding (beam_size=1)
      }

    Raises:
      NotImplementedError: If there are multiple data shards.
    """
    # For real-valued modalities use the slow decode path for now.
    if (self._target_modality_is_real or
        self._hparams.self_attention_type != "dot_product"):
      return  super(Transformer, self)._greedy_infer(features, decode_length)
    with tf.variable_scope(self.name):
      if use_tpu:
        return self._fast_decode_tpu(features, decode_length)
      return self._fast_decode(features, decode_length)

  def _beam_decode(self,
                   features,
                   decode_length,
                   beam_size,
                   top_beams,
                   alpha,
                   use_tpu=False):
    """Beam search decoding.

    Args:
      features: an map of string to `Tensor`
      decode_length: an integer.  How many additional timesteps to decode.
      beam_size: number of beams.
      top_beams: an integer. How many of the beams to return.
      alpha: Float that controls the length penalty. larger the alpha, stronger
        the preference for longer translations.
      use_tpu: A bool, whether to do beam decode on TPU.

    Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, <= decode_length] if beam_size == 1 or
              [batch_size, top_beams, <= decode_length]
          "scores": decoding log probs from the beam search,
              None if using greedy decoding (beam_size=1)
      }
    """
    if (self._hparams.self_attention_type not in ["dot_product",
                                                  "dot_product_relative"]):
      # Caching is not guaranteed to work with attention types other than
      # dot_product.
      # TODO(petershaw): Support fast decoding when using relative
      # position representations, i.e. "dot_product_relative" attention.
      return self._beam_decode_slow(features, decode_length, beam_size,
                                    top_beams, alpha, use_tpu)
    with tf.variable_scope(self.name):
      if use_tpu:
        return self._fast_decode_tpu(
            features, decode_length, beam_size, top_beams, alpha)
      return self._fast_decode(
          features, decode_length, beam_size, top_beams, alpha)

  def _fast_decode_tpu(self,
                       features,
                       decode_length,
                       beam_size=1,
                       top_beams=1,
                       alpha=1.0):
    """Fast decoding.

    Implements both greedy and beam search decoding on TPU, uses beam search
    iff beam_size > 1, otherwise beam search related arguments are ignored.

    Args:
      features: A map of string to model features.
      decode_length: An integer, how many additional timesteps to decode.
      beam_size: An integer, number of beams.
      top_beams: An integer, how many of the beams to return.
      alpha: A float that controls the length penalty. Larger the alpha,
        stronger the preference for longer translations.

    Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, <= decode_length] if beam_size == 1 or
              [batch_size, top_beams, <= decode_length]
          "scores": decoding log probs from the beam search,
              None if using greedy decoding (beam_size=1)
      }.

    Raises:
      NotImplementedError: If there are multiple data shards.
    """
    if self._num_datashards != 1:
      raise NotImplementedError("Fast decoding only supports a single shard.")
    if "targets_segmentation" in features:
      raise NotImplementedError(
          "Decoding not supported on packed datasets "
          " If you want to decode from a dataset, use the non-packed version"
          " of the dataset when decoding.")
    dp = self._data_parallelism
    hparams = self._hparams
    target_modality = self._problem_hparams.modality["targets"]

    if self.has_input:
      inputs = features["inputs"]
      if target_modality.is_class_modality:
        decode_length = 1
      else:
        decode_length = (
            common_layers.shape_list(inputs)[1] + features.get(
                "decode_length", decode_length))

      # TODO(llion): Clean up this reshaping logic.
      inputs = tf.expand_dims(inputs, axis=1)
      if len(inputs.shape) < 5:
        inputs = tf.expand_dims(inputs, axis=4)
      s = common_layers.shape_list(inputs)
      batch_size = s[0]
      inputs = tf.reshape(inputs, [s[0] * s[1], s[2], s[3], s[4]])
      # _shard_features called to ensure that the variable names match
      inputs = self._shard_features({"inputs": inputs})["inputs"]
      input_modality = self._problem_hparams.modality["inputs"]
      with tf.variable_scope(input_modality.name):
        inputs = input_modality.bottom_sharded(inputs, dp)
      with tf.variable_scope("body"):
        encoder_output, encoder_decoder_attention_bias = dp(
            self.encode,
            inputs,
            features["target_space_id"],
            hparams,
            features=features)
      encoder_output = encoder_output[0]
      encoder_decoder_attention_bias = encoder_decoder_attention_bias[0]
      partial_targets = None
    else:
      # The problem has no inputs.
      encoder_output = None
      encoder_decoder_attention_bias = None

      # Prepare partial targets.
      # In either features["inputs"] or features["targets"].
      # We force the outputs to begin with these sequences.
      partial_targets = features.get("inputs")
      if partial_targets is None:
        partial_targets = features["targets"]
      assert partial_targets is not None
      partial_targets = common_layers.expand_squeeze_to_nd(partial_targets, 2)
      partial_targets = tf.to_int64(partial_targets)
      partial_targets_shape = common_layers.shape_list(partial_targets)
      partial_targets_length = partial_targets_shape[1]
      decode_length = (
          partial_targets_length + features.get("decode_length", decode_length))
      batch_size = partial_targets_shape[0]

    if hparams.pos == "timing":
      positional_encoding = common_attention.get_timing_signal_1d(
          decode_length + 1, hparams.hidden_size)
    elif hparams.pos == "emb":
      positional_encoding = common_attention.add_positional_embedding(
          tf.zeros([1, decode_length + 1, hparams.hidden_size]),
          hparams.max_length, "body/targets_positional_embedding", None)
    else:
      positional_encoding = None

    def preprocess_targets(targets, i):
      """Performs preprocessing steps on the targets to prepare for the decoder.

      This includes:
        - Embedding the ids.
        - Flattening to 3D tensor.
        - Optionally adding timing signals.

      Args:
        targets: A tensor, inputs ids to the decoder. [batch_size, 1].
        i: An integer, Step number of the decoding loop.

      Returns:
        A tensor, processed targets [batch_size, 1, hidden_dim].
      """
      # _shard_features called to ensure that the variable names match
      targets = self._shard_features({"targets": targets})["targets"]
      with tf.variable_scope(target_modality.name):
        targets = target_modality.targets_bottom_sharded(targets, dp)[0]
      targets = common_layers.flatten4d3d(targets)

      # TODO(llion): Explain! Is this even needed?
      targets = tf.cond(
          tf.equal(i, 0), lambda: tf.zeros_like(targets), lambda: targets)

      if positional_encoding is not None:
        positional_encoding_shape = positional_encoding.shape.as_list()
        targets += tf.slice(
            positional_encoding, [0, i, 0],
            [positional_encoding_shape[0], 1, positional_encoding_shape[2]])
      return targets

    decoder_self_attention_bias = (
        common_attention.attention_bias_lower_triangle(decode_length))
    if hparams.proximity_bias:
      decoder_self_attention_bias += common_attention.attention_bias_proximal(
          decode_length)

    def symbols_to_logits_tpu_fn(ids, i, cache):
      """Go from ids to logits for next symbol on TPU.

      Args:
        ids: A tensor, symbol IDs.
        i: An integer, step number of the decoding loop. Only used for inference
            on TPU.
        cache: A dict, containing tensors which are the results of previous
            attentions, used for fast decoding.

      Returns:
        ret: A tensor, computed logits.
        cache: A dict, containing tensors which are the results of previous
            attentions, used for fast decoding.
      """
      ids = ids[:, -1:]
      targets = tf.expand_dims(tf.expand_dims(ids, axis=2), axis=3)
      targets = preprocess_targets(targets, i)

      bias_shape = decoder_self_attention_bias.shape.as_list()
      bias = tf.slice(decoder_self_attention_bias, [0, 0, i, 0],
                      [bias_shape[0], bias_shape[1], 1, bias_shape[3]])

      with tf.variable_scope("body"):
        body_outputs = dp(
            self.decode,
            targets,
            cache.get("encoder_output"),
            cache.get("encoder_decoder_attention_bias"),
            bias,
            hparams,
            cache,
            i,
            nonpadding=features_to_nonpadding(features, "targets"))

      with tf.variable_scope(target_modality.name):
        logits = target_modality.top_sharded(body_outputs, None, dp)[0]

      ret = tf.squeeze(logits, axis=[1, 2, 3])
      if partial_targets is not None:
        # If the position is within the given partial targets, we alter the
        # logits to always return those values.
        # A faster approach would be to process the partial targets in one
        # iteration in order to fill the corresponding parts of the cache.
        # This would require broader changes, though.
        vocab_size = tf.shape(ret)[1]

        def forced_logits():
          return tf.one_hot(
              tf.tile(
                  tf.slice(partial_targets, [0, i],
                           [partial_targets.shape.as_list()[0], 1]),
                  [beam_size]), vocab_size, 0.0, -1e9)

        ret = tf.cond(
            tf.less(i, partial_targets_length), forced_logits, lambda: ret)
      return ret, cache

    ret = fast_decode_tpu(
        encoder_output=encoder_output,
        encoder_decoder_attention_bias=encoder_decoder_attention_bias,
        symbols_to_logits_fn=symbols_to_logits_tpu_fn,
        hparams=hparams,
        decode_length=decode_length,
        vocab_size=target_modality.top_dimensionality,
        beam_size=beam_size,
        top_beams=top_beams,
        alpha=alpha,
        batch_size=batch_size,
        force_decode_length=self._decode_hparams.force_decode_length)
    if partial_targets is not None:
      if beam_size <= 1 or top_beams <= 1:
        ret["outputs"] = ret["outputs"][:, partial_targets_length:]
      else:
        ret["outputs"] = ret["outputs"][:, :, partial_targets_length:]
    return ret

  def _fast_decode(self,
                   features,
                   decode_length,
                   beam_size=1,
                   top_beams=1,
                   alpha=1.0):
    """Fast decoding.

    Implements both greedy and beam search decoding, uses beam search iff
    beam_size > 1, otherwise beam search related arguments are ignored.

    Args:
      features: a map of string to model  features.
      decode_length: an integer.  How many additional timesteps to decode.
      beam_size: number of beams.
      top_beams: an integer. How many of the beams to return.
      alpha: Float that controls the length penalty. larger the alpha, stronger
        the preference for longer translations.

    Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, <= decode_length] if beam_size == 1 or
              [batch_size, top_beams, <= decode_length]
          "scores": decoding log probs from the beam search,
              None if using greedy decoding (beam_size=1)
      }

    Raises:
      NotImplementedError: If there are multiple data shards.
    """
    if self._num_datashards != 1:
      raise NotImplementedError("Fast decoding only supports a single shard.")
    dp = self._data_parallelism
    hparams = self._hparams
    target_modality = self._problem_hparams.modality["targets"]
    if "targets_segmentation" in features:
      raise NotImplementedError(
          "Decoding not supported on packed datasets "
          " If you want to decode from a dataset, use the non-packed version"
          " of the dataset when decoding.")
    if self.has_input:
      inputs = features["inputs"]
      if target_modality.is_class_modality:
        decode_length = 1
      else:
        decode_length = (
            common_layers.shape_list(inputs)[1] + features.get(
                "decode_length", decode_length))

      # TODO(llion): Clean up this reshaping logic.
      inputs = tf.expand_dims(inputs, axis=1)
      if len(inputs.shape) < 5:
        inputs = tf.expand_dims(inputs, axis=4)
      s = common_layers.shape_list(inputs)
      batch_size = s[0]
      inputs = tf.reshape(inputs, [s[0] * s[1], s[2], s[3], s[4]])
      # _shard_features called to ensure that the variable names match
      inputs = self._shard_features({"inputs": inputs})["inputs"]
      input_modality = self._problem_hparams.modality["inputs"]
      with tf.variable_scope(input_modality.name):
        inputs = input_modality.bottom_sharded(inputs, dp)
      with tf.variable_scope("body"):
        encoder_output, encoder_decoder_attention_bias = dp(
            self.encode,
            inputs,
            features["target_space_id"],
            hparams,
            features=features)
      encoder_output = encoder_output[0]
      encoder_decoder_attention_bias = encoder_decoder_attention_bias[0]
      partial_targets = None
    else:
      # The problem has no inputs.
      encoder_output = None
      encoder_decoder_attention_bias = None

      # Prepare partial targets.
      # In either features["inputs"] or features["targets"].
      # We force the outputs to begin with these sequences.
      partial_targets = features.get("inputs")
      if partial_targets is None:
        partial_targets = features["targets"]
      assert partial_targets is not None
      partial_targets = common_layers.expand_squeeze_to_nd(partial_targets, 2)
      partial_targets = tf.to_int64(partial_targets)
      partial_targets_shape = common_layers.shape_list(partial_targets)
      partial_targets_length = partial_targets_shape[1]
      decode_length = (
          partial_targets_length + features.get("decode_length", decode_length))
      batch_size = partial_targets_shape[0]

    if hparams.pos == "timing":
      positional_encoding = common_attention.get_timing_signal_1d(
          decode_length + 1, hparams.hidden_size)
    elif hparams.pos == "emb":
      positional_encoding = common_attention.add_positional_embedding(
          tf.zeros([1, decode_length, hparams.hidden_size]),
          hparams.max_length, "body/targets_positional_embedding", None)
    else:
      positional_encoding = None

    def preprocess_targets(targets, i):
      """Performs preprocessing steps on the targets to prepare for the decoder.

      This includes:
        - Embedding the ids.
        - Flattening to 3D tensor.
        - Optionally adding timing signals.

      Args:
        targets: inputs ids to the decoder. [batch_size, 1]
        i: scalar, Step number of the decoding loop.

      Returns:
        Processed targets [batch_size, 1, hidden_dim]
      """
      # _shard_features called to ensure that the variable names match
      targets = self._shard_features({"targets": targets})["targets"]
      with tf.variable_scope(target_modality.name):
        targets = target_modality.targets_bottom_sharded(targets, dp)[0]
      targets = common_layers.flatten4d3d(targets)

      # TODO(llion): Explain! Is this even needed?
      targets = tf.cond(
          tf.equal(i, 0), lambda: tf.zeros_like(targets), lambda: targets)

      if positional_encoding is not None:
        targets += positional_encoding[:, i:i + 1]
      return targets

    decoder_self_attention_bias = (
        common_attention.attention_bias_lower_triangle(decode_length))
    if hparams.proximity_bias:
      decoder_self_attention_bias += common_attention.attention_bias_proximal(
          decode_length)

    def symbols_to_logits_fn(ids, i, cache):
      """Go from ids to logits for next symbol."""
      ids = ids[:, -1:]
      targets = tf.expand_dims(tf.expand_dims(ids, axis=2), axis=3)
      targets = preprocess_targets(targets, i)

      bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]

      with tf.variable_scope("body"):
        body_outputs = dp(
            self.decode,
            targets,
            cache.get("encoder_output"),
            cache.get("encoder_decoder_attention_bias"),
            bias,
            hparams,
            cache,
            nonpadding=features_to_nonpadding(features, "targets"))

      with tf.variable_scope(target_modality.name):
        logits = target_modality.top_sharded(body_outputs, None, dp)[0]

      ret = tf.squeeze(logits, axis=[1, 2, 3])
      if partial_targets is not None:
        # If the position is within the given partial targets, we alter the
        # logits to always return those values.
        # A faster approach would be to process the partial targets in one
        # iteration in order to fill the corresponding parts of the cache.
        # This would require broader changes, though.
        vocab_size = tf.shape(ret)[1]

        def forced_logits():
          return tf.one_hot(
              tf.tile(partial_targets[:, i], [beam_size]), vocab_size, 0.0,
              -1e9)

        ret = tf.cond(
            tf.less(i, partial_targets_length), forced_logits, lambda: ret)
      return ret, cache

    print("Vocab Size={}".format(target_modality.top_dimensionality))
    ret = fast_decode(
        encoder_output=encoder_output,
        encoder_decoder_attention_bias=encoder_decoder_attention_bias,
        symbols_to_logits_fn=symbols_to_logits_fn,
        hparams=hparams,
        decode_length=decode_length,
        vocab_size=target_modality.top_dimensionality,
        beam_size=beam_size,
        top_beams=top_beams,
        alpha=alpha,
        batch_size=batch_size,
        force_decode_length=self._decode_hparams.force_decode_length)
    if partial_targets is not None:
      if beam_size <= 1 or top_beams <= 1:
        ret["outputs"] = ret["outputs"][:, partial_targets_length:]
      else:
        ret["outputs"] = ret["outputs"][:, :, partial_targets_length:]
    return ret


def fast_decode_tpu(encoder_output,
                    encoder_decoder_attention_bias,
                    symbols_to_logits_fn,
                    hparams,
                    decode_length,
                    vocab_size,
                    beam_size=1,
                    top_beams=1,
                    alpha=1.0,
                    sos_id=0,
                    eos_id=beam_search.EOS_ID,
                    batch_size=None,
                    force_decode_length=False,
                    scope_prefix="body/"):
  """Given encoder output and a symbols to logits function, does fast decoding.

  Implements both greedy and beam search decoding for TPU, uses beam search iff
  beam_size > 1, otherwise beam search related arguments are ignored.

  Args:
    encoder_output: A tensor, output from encoder.
    encoder_decoder_attention_bias: A tensor, bias for use in encoder-decoder
        attention.
    symbols_to_logits_fn: Incremental decoding, function mapping triple
        `(ids, step, cache)` to symbol logits.
    hparams: Run hyperparameters.
    decode_length: An integer, how many additional timesteps to decode.
    vocab_size: Output vocabulary size.
    beam_size: An integer, number of beams.
    top_beams: An integer, how many of the beams to return.
    alpha: A float that controls the length penalty. Larger the alpha, stronger
      the preference for longer translations.
    sos_id: Start-of-sequence symbol.
    eos_id: End-of-sequence symbol.
    batch_size: An integer, must be passed if there is no input.
    force_decode_length: A bool, whether to force the full decode length, or if
        False, stop when all beams hit eos_id.
    scope_prefix: str, prefix for decoder layer variable scopes.

  Returns:
    A dict of decoding results {
        "outputs": integer `Tensor` of decoded ids of shape
            [batch_size, <= decode_length] if top_beams == 1 or
            [batch_size, top_beams, <= decode_length] otherwise
        "scores": decoding log probs from the beam search,
            None if using greedy decoding (beam_size=1)
    }.

  Raises:
    NotImplementedError: If beam size > 1 with partial targets.
  """
  print(encoder_output)
  if encoder_output is not None:
    batch_size = common_layers.shape_list(encoder_output)[0]

  if(hparams.light_mode=='ftm'):
      key_channels = hparams.mini_dim * hparams.num_heads
  key_channels = hparams.attention_key_channels or hparams.hidden_size
  value_channels = hparams.attention_value_channels or hparams.hidden_size
  num_layers = hparams.num_decoder_layers or hparams.num_hidden_layers
  vars_3d_num_heads = (
      hparams.num_heads if hparams.get("attention_variables_3d") else 0)

  cache = {
      "layer_%d" % layer: {
          "k":
          common_attention.split_heads(
              tf.zeros([batch_size, decode_length, key_channels]),
              hparams.num_heads),
          "v":
          common_attention.split_heads(
              tf.zeros([batch_size, decode_length, value_channels]),
              hparams.num_heads),
          "f":
          tf.zeros([batch_size, decode_length, hparams.hidden_size]),
      } for layer in range(num_layers)
  }

  if encoder_output is not None:
    for layer in range(num_layers):
      layer_name = "layer_%d" % layer
      with tf.variable_scope(
          "%sdecoder/%s/encdec_attention/multihead_attention" % (scope_prefix,
                                                                 layer_name)):
        k_encdec = common_attention.compute_attention_component(
            encoder_output, key_channels, name="k",
            vars_3d_num_heads=vars_3d_num_heads, mode=hparams.light_mode, factor=hparams.factor)
        k_encdec = common_attention.split_heads(k_encdec, hparams.num_heads, light_mode=hparams.light_mode
                                    )
        v_encdec = common_attention.compute_attention_component(
            encoder_output, value_channels, name="v",
            vars_3d_num_heads=vars_3d_num_heads, mode=hparams.light_mode, factor=hparams.factor)
        v_encdec = common_attention.split_heads(v_encdec, hparams.num_heads, light_mode=hparams.light_mode)
      cache[layer_name]["k_encdec"] = k_encdec
      cache[layer_name]["v_encdec"] = v_encdec

    cache["encoder_output"] = encoder_output
    cache["encoder_decoder_attention_bias"] = encoder_decoder_attention_bias

  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_SEQ_BEAM_SEARCH,
      value={
          "vocab_size": vocab_size,
          "batch_size": batch_size,
          "beam_size": beam_size,
          "alpha": alpha,
          "max_decode_length": decode_length
      })
  if beam_size > 1:  # Beam Search
    initial_ids = sos_id * tf.ones([batch_size], dtype=tf.int32)
    decoded_ids = beam_search.beam_search(
        symbols_to_logits_fn,
        initial_ids,
        beam_size,
        decode_length,
        vocab_size,
        alpha,
        states=cache,
        eos_id=eos_id,
        stop_early=(top_beams == 1),
        use_tpu=True)

    if top_beams == 1:
      decoded_ids = decoded_ids[:, 0, 1:]
      scores = scores[:, 0]
    else:
      decoded_ids = decoded_ids[:, :top_beams, 1:]
      scores = scores[:, :top_beams]
  else:  # Greedy
    def inner_loop(i, hit_eos, next_id, decoded_ids, cache, log_prob):
      """One step of greedy decoding."""
      logits, cache = symbols_to_logits_fn(next_id, i, cache)
      log_probs = common_layers.log_prob_from_logits(logits)
      temperature = getattr(hparams, "sampling_temp", 0.0)
      if hparams.sampling_method == "argmax":
        temperature = 0.0
      next_id = common_layers.sample_with_temperature(logits, temperature)
      hit_eos |= tf.equal(next_id, eos_id)

      log_prob_indices = tf.stack(
          [tf.range(tf.to_int64(batch_size)), next_id], axis=1)
      log_prob += tf.gather_nd(log_probs, log_prob_indices)

      next_id = tf.expand_dims(next_id, axis=1)
      decoded_ids = tf.transpose(decoded_ids)
      decoded_ids = inplace_ops.alias_inplace_update(
          decoded_ids, i, tf.squeeze(next_id, axis=1))
      decoded_ids = tf.transpose(decoded_ids)
      return i + 1, hit_eos, next_id, decoded_ids, cache, log_prob

    def is_not_finished(i, hit_eos, *_):
      finished = i >= decode_length
      if not force_decode_length:
        finished |= tf.reduce_all(hit_eos)
      return tf.logical_not(finished)

    decoded_ids = tf.zeros([batch_size, decode_length], dtype=tf.int64)
    hit_eos = tf.fill([batch_size], False)
    next_id = sos_id * tf.ones([batch_size, 1], dtype=tf.int64)
    initial_log_prob = tf.zeros([batch_size], dtype=tf.float32)

    def compute_cache_shape_invariants(tensor):
      return tf.TensorShape(tensor.shape.as_list())

    _, _, _, decoded_ids, _, log_prob = tf.while_loop(
        is_not_finished,
        inner_loop, [
            tf.constant(0), hit_eos, next_id, decoded_ids, cache,
            initial_log_prob
        ],
        shape_invariants=[
            tf.TensorShape([]),
            tf.TensorShape([batch_size]),
            tf.TensorShape([batch_size, 1]),
            tf.TensorShape([batch_size, decode_length]),
            nest.map_structure(compute_cache_shape_invariants, cache),
            tf.TensorShape([batch_size]),
        ])
    scores = log_prob

  return {"outputs": decoded_ids, "scores": scores}


def fast_decode(encoder_output,
                encoder_decoder_attention_bias,
                symbols_to_logits_fn,
                hparams,
                decode_length,
                vocab_size,
                beam_size=1,
                top_beams=1,
                alpha=1.0,
                sos_id=0,
                eos_id=beam_search.EOS_ID,
                batch_size=None,
                force_decode_length=False,
                scope_prefix="body/"):
  """Given encoder output and a symbols to logits function, does fast decoding.

  Implements both greedy and beam search decoding, uses beam search iff
  beam_size > 1, otherwise beam search related arguments are ignored.

  Args:
    encoder_output: Output from encoder.
    encoder_decoder_attention_bias: a bias tensor for use in encoder-decoder
      attention
    symbols_to_logits_fn: Incremental decoding; function mapping triple
      `(ids, step, cache)` to symbol logits.
    hparams: run hyperparameters
    decode_length: an integer.  How many additional timesteps to decode.
    vocab_size: Output vocabulary size.
    beam_size: number of beams.
    top_beams: an integer. How many of the beams to return.
    alpha: Float that controls the length penalty. larger the alpha, stronger
      the preference for longer translations.
    sos_id: End-of-sequence symbol in beam search.
    eos_id: End-of-sequence symbol in beam search.
    batch_size: an integer scalar - must be passed if there is no input
    force_decode_length: bool, whether to force the full decode length, or if
      False, stop when all beams hit eos_id.
    scope_prefix: str, prefix for decoder layer variable scopes.

  Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, <= decode_length] if top_beams == 1 or
              [batch_size, top_beams, <= decode_length] otherwise
          "scores": decoding log probs from the beam search,
              None if using greedy decoding (beam_size=1)
      }

    Raises:
      NotImplementedError: If beam size > 1 with partial targets.
  """
  if encoder_output is not None:
    batch_size = common_layers.shape_list(encoder_output)[0]

  key_channels = hparams.attention_key_channels or hparams.hidden_size
  if(hparams.light_mode=='ftm'):
      key_channels = hparams.mini_dim * hparams.num_heads
  value_channels = hparams.attention_value_channels or hparams.hidden_size
  num_layers = hparams.num_decoder_layers or hparams.num_hidden_layers
  vars_3d_num_heads = (
      hparams.num_heads if hparams.get("attention_variables_3d") else 0)

  if(hparams.upsize!=0):
      # print("[Old] Key={} Value={}".format(total_key_depth, total_value_depth))
      key_channels *= hparams.upsize
      value_channels *= hparams.upsize
      key_channels = int(key_channels)
      value_channels = int(value_channels)
      # print("[Up] Key={} Value={}".format(total_key_depth, total_value_depth))

  cache = {
      "layer_%d" % layer: {
          "k":
              common_attention.split_heads(
                  tf.zeros([batch_size, 0, key_channels]), hparams.num_heads, light_mode=hparams.light_mode),
          "v":
              common_attention.split_heads(
                  tf.zeros([batch_size, 0, value_channels]), hparams.num_heads, light_mode=hparams.light_mode),
          "f":
              tf.zeros([batch_size, 0, hparams.hidden_size]),
      } for layer in range(num_layers)
  }

  if encoder_output is not None:
    for layer in range(num_layers):
      layer_name = "layer_%d" % layer
      with tf.variable_scope(
          "%sdecoder/%s/encdec_attention/multihead_attention" % (scope_prefix,
                                                                 layer_name)):
        k_encdec = common_attention.compute_attention_component(
            encoder_output, key_channels, name="k",
            vars_3d_num_heads=vars_3d_num_heads, mode=hparams.light_mode, factor=hparams.factor)
        k_encdec = common_attention.split_heads(k_encdec, hparams.num_heads, light_mode=hparams.light_mode)
        v_encdec = common_attention.compute_attention_component(
            encoder_output, value_channels, name="v",
            vars_3d_num_heads=vars_3d_num_heads, mode=hparams.light_mode, factor=hparams.factor)
        v_encdec = common_attention.split_heads(v_encdec, hparams.num_heads, light_mode=hparams.light_mode)
      cache[layer_name]["k_encdec"] = k_encdec
      cache[layer_name]["v_encdec"] = v_encdec

    cache["encoder_output"] = encoder_output
    cache["encoder_decoder_attention_bias"] = encoder_decoder_attention_bias

  if beam_size > 1:  # Beam Search
    initial_ids = sos_id * tf.ones([batch_size], dtype=tf.int32)
    decoded_ids, scores, cache = beam_search.beam_search(
        symbols_to_logits_fn,
        initial_ids,
        beam_size,
        decode_length,
        vocab_size,
        alpha,
        states=cache,
        eos_id=eos_id,
        stop_early=(top_beams == 1))

    if top_beams == 1:
      decoded_ids = decoded_ids[:, 0, 1:]
      scores = scores[:, 0]
    else:
      decoded_ids = decoded_ids[:, :top_beams, 1:]
      scores = scores[:, :top_beams]
  else:  # Greedy

    def inner_loop(i, hit_eos, next_id, decoded_ids, cache, log_prob):
      """One step of greedy decoding."""
      logits, cache = symbols_to_logits_fn(next_id, i, cache)
      log_probs = common_layers.log_prob_from_logits(logits)
      temperature = getattr(hparams, "sampling_temp", 0.0)
      if hparams.sampling_method == "argmax":
        temperature = 0.0
      next_id = common_layers.sample_with_temperature(logits, temperature)
      hit_eos |= tf.equal(next_id, eos_id)

      log_prob_indices = tf.stack(
          [tf.range(tf.to_int64(batch_size)), next_id], axis=1)
      log_prob += tf.gather_nd(log_probs, log_prob_indices)

      next_id = tf.expand_dims(next_id, axis=1)
      decoded_ids = tf.concat([decoded_ids, next_id], axis=1)
      return i + 1, hit_eos, next_id, decoded_ids, cache, log_prob

    def is_not_finished(i, hit_eos, *_):
      finished = i >= decode_length
      if not force_decode_length:
        finished |= tf.reduce_all(hit_eos)
      return tf.logical_not(finished)

    decoded_ids = tf.zeros([batch_size, 0], dtype=tf.int64)
    hit_eos = tf.fill([batch_size], False)
    next_id = sos_id * tf.ones([batch_size, 1], dtype=tf.int64)
    initial_log_prob = tf.zeros([batch_size], dtype=tf.float32)
    _, _, _, decoded_ids, _, log_prob = tf.while_loop(
        is_not_finished,
        inner_loop, [
            tf.constant(0), hit_eos, next_id, decoded_ids, cache,
            initial_log_prob
        ],
        shape_invariants=[
            tf.TensorShape([]),
            tf.TensorShape([None]),
            tf.TensorShape([None, None]),
            tf.TensorShape([None, None]),
            nest.map_structure(beam_search.get_state_shape_invariants, cache),
            tf.TensorShape([None]),
        ])
    scores = log_prob

  return {"outputs": decoded_ids, "scores": scores}



@registry.register_model
class lightTransformerEncoder(t2t_model.T2TModel):
  """Transformer, encoder only."""

  def body(self, features):
    hparams = self._hparams
    inputs = features["inputs"]
    target_space = features["target_space_id"]

    inputs = common_layers.flatten4d3d(inputs)

    (encoder_input, encoder_self_attention_bias, _) = (
        transformer_prepare_encoder(inputs, target_space, hparams))

    encoder_input = tf.nn.dropout(encoder_input,
                                  1.0 - hparams.layer_prepostprocess_dropout)
    encoder_output = transformer_encoder(
        encoder_input,
        encoder_self_attention_bias,
        hparams,
        nonpadding=features_to_nonpadding(features, "inputs"))
    encoder_output = tf.expand_dims(encoder_output, 2)

    return encoder_output


def features_to_nonpadding(features, inputs_or_targets="inputs"):
  key = inputs_or_targets + "_segmentation"
  if features and key in features:
    return tf.minimum(tf.to_float(features[key]), 1.0)
  return None


def transformer_prepare_decoder(targets, hparams, features=None):
  """Prepare one shard of the model for the decoder.

  Args:
    targets: a Tensor.
    hparams: run hyperparameters
    features: optionally pass the entire features dictionary as well.
      This is needed now for "packed" datasets.

  Returns:
    decoder_input: a Tensor, bottom of decoder stack
    decoder_self_attention_bias: a bias tensor for use in decoder self-attention
  """
  if hparams.causal_decoder_self_attention:
    # Causal attention.
    if hparams.prepend_mode == "prepend_inputs_full_attention":
      decoder_self_attention_bias = (
          common_attention.attention_bias_prepend_inputs_full_attention(
              common_attention.embedding_to_padding(targets)))
    else:
      decoder_self_attention_bias = (
          common_attention.attention_bias_lower_triangle(
              common_layers.shape_list(targets)[1]))
  else:
    # Full attention.
    decoder_padding = common_attention.embedding_to_padding(targets)
    decoder_self_attention_bias = (
        common_attention.attention_bias_ignore_padding(decoder_padding))

  if features and "targets_segmentation" in features:
    # "Packed" dataset - keep the examples from seeing each other.
    targets_segmentation = features["targets_segmentation"]
    targets_position = features["targets_position"]
    decoder_self_attention_bias += common_attention.attention_bias_same_segment(
        targets_segmentation, targets_segmentation)
  else:
    targets_position = None
  if hparams.proximity_bias:
    decoder_self_attention_bias += common_attention.attention_bias_proximal(
        common_layers.shape_list(targets)[1])
  decoder_input = common_layers.shift_right_3d(targets)
  if hparams.pos == "timing":
    if targets_position is not None:
      decoder_input = common_attention.add_timing_signal_1d_given_position(
          decoder_input, targets_position)
    else:
      decoder_input = common_attention.add_timing_signal_1d(decoder_input)
  elif hparams.pos == "emb":
    decoder_input = common_attention.add_positional_embedding(
        decoder_input, hparams.max_length, "targets_positional_embedding",
        targets_position)

  if hparams.activation_dtype == "bfloat16":
    decoder_self_attention_bias = tf.cast(decoder_self_attention_bias,
                                          tf.bfloat16)
  return (decoder_input, decoder_self_attention_bias)


def transformer_decoder(decoder_input,
                        encoder_output,
                        decoder_self_attention_bias,
                        encoder_decoder_attention_bias,
                        hparams,
                        cache=None,
                        decode_loop_step=None,
                        name="decoder",
                        nonpadding=None,
                        save_weights_to=None,
                        make_image_summary=True,
                        losses=None):
  """A stack of transformer layers.

  Args:
    decoder_input: a Tensor
    encoder_output: a Tensor
    decoder_self_attention_bias: bias Tensor for self-attention
      (see common_attention.attention_bias())
    encoder_decoder_attention_bias: bias Tensor for encoder-decoder attention
      (see common_attention.attention_bias())
    hparams: hyperparameters for model
    cache: dict, containing tensors which are the results of previous
        attentions, used for fast decoding.
    decode_loop_step: An integer, step number of the decoding loop.
        Only used for inference on TPU.
    name: a string
    nonpadding: optional Tensor with shape [batch_size, encoder_length]
      indicating what positions are not padding.  This is used
      to mask out padding in convolutional layers.  We generally only
      need this mask for "packed" datasets, because for ordinary datasets,
      no padding is ever followed by nonpadding.
    save_weights_to: an optional dictionary to capture attention weights
      for visualization; the weights tensor will be appended there under
      a string key created from the variable scope (including name).
    make_image_summary: Whether to make an attention image summary.
    losses: optional list onto which to append extra training losses

  Returns:
    y: a Tensors
  """
  x = decoder_input
  attention_dropout_broadcast_dims = (
      common_layers.comma_separated_string_to_integer_list(
          getattr(hparams, "attention_dropout_broadcast_dims", "")))

  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_NUM_HIDDEN_LAYERS,
      value=hparams.num_decoder_layers or hparams.num_hidden_layers)
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
  is_training = hparams.mode == tf.contrib.learn.ModeKeys.TRAIN
  res1, res2 = None, None

  with tf.variable_scope(name):
    for layer in range(hparams.num_decoder_layers or hparams.num_hidden_layers):
      layer_name = "layer_%d" % layer
      layer_cache = cache[layer_name] if cache is not None else None
      with tf.variable_scope(layer_name):
        with tf.variable_scope("self_attention"):
          y, res1 = common_attention.multihead_attention(
              common_layers.layer_preprocess(x, hparams),
              None,
              decoder_self_attention_bias,
              hparams.attention_key_channels or hparams.hidden_size,
              hparams.attention_value_channels or hparams.hidden_size,
              hparams.hidden_size,
              hparams.num_heads,
              hparams.attention_dropout,
              attention_type=hparams.dec_attention_type,
              max_relative_position=hparams.max_relative_position,
              heads_share_relative_embedding=(
                  hparams.heads_share_relative_embedding),
              add_relative_to_values=hparams.add_relative_to_values,
              save_weights_to=save_weights_to,
              cache=layer_cache,
              make_image_summary=make_image_summary,
              dropout_broadcast_dims=attention_dropout_broadcast_dims,
              max_length=hparams.get("max_length"),
              decode_loop_step=decode_loop_step,
              vars_3d=hparams.get("attention_variables_3d"),
              is_training=is_training,
              light_mode=hparams.light_mode,
              hparams=hparams,
              att_mode='self_causal',
              res=res1)
          x = common_layers.layer_postprocess(x, y, hparams)
        if encoder_output is not None:
          with tf.variable_scope("encdec_attention"):
            y, res2 = common_attention.multihead_attention(
                common_layers.layer_preprocess(x, hparams),
                encoder_output,
                encoder_decoder_attention_bias,
                hparams.attention_key_channels or hparams.hidden_size,
                hparams.attention_value_channels or hparams.hidden_size,
                hparams.hidden_size,
                hparams.num_heads,
                hparams.attention_dropout,
                max_relative_position=hparams.max_relative_position,
                attention_type=hparams.encdec_attention_type,
                heads_share_relative_embedding=(
                    hparams.heads_share_relative_embedding),
                add_relative_to_values=hparams.add_relative_to_values,
                save_weights_to=save_weights_to,
                cache=layer_cache,
                make_image_summary=make_image_summary,
                dropout_broadcast_dims=attention_dropout_broadcast_dims,
                max_length=hparams.get("max_length"),
                vars_3d=hparams.get("attention_variables_3d"),
                is_training=is_training,
                light_mode=hparams.light_mode,
                hparams=hparams,
                att_mode='encdec',
                res=res2)
            x = common_layers.layer_postprocess(x, y, hparams)
        with tf.variable_scope("ffn"):
          y = transformer_ffn_layer(
              common_layers.layer_preprocess(x, hparams),
              hparams,
              conv_padding="LEFT",
              nonpadding_mask=nonpadding,
              losses=losses,
              cache=layer_cache,
              decode_loop_step=decode_loop_step)
          x = common_layers.layer_postprocess(x, y, hparams)
    # if normalization is done in layer_preprocess, then it should also be done
    # on the output, since the output can grow very large, being the sum of
    # a whole stack of unnormalized layer outputs.
    mlperf_log.transformer_print(
        key=mlperf_log.MODEL_HP_NORM,
        value={"hidden_size": hparams.hidden_size})
    return common_layers.layer_preprocess(x, hparams)


@registry.register_hparams
def light_transformer_base_v1():
  """Set of hyperparameters."""
  hparams = common_hparams.basic_params1()
  hparams.norm_type = "layer"
  hparams.hidden_size = 512
  hparams.batch_size = 4096
  hparams.max_length = 256
  hparams.clip_grad_norm = 0.  # i.e. no gradient clipping
  hparams.optimizer_adam_epsilon = 1e-9
  hparams.learning_rate_schedule = "legacy"
  hparams.learning_rate_decay_scheme = "noam"
  hparams.learning_rate = 0.1
  hparams.learning_rate_warmup_steps = 4000
  hparams.initializer_gain = 1.0
  hparams.num_hidden_layers = 6
  hparams.initializer = "uniform_unit_scaling"
  hparams.weight_decay = 0.0
  hparams.optimizer_adam_beta1 = 0.9
  hparams.optimizer_adam_beta2 = 0.98
  hparams.num_sampled_classes = 0
  hparams.label_smoothing = 0.1
  hparams.shared_embedding_and_softmax_weights = True
  hparams.symbol_modality_num_shards = 16
  hparams.max_relative_position = 20

  # Add new ones like this.
  hparams.add_hparam("filter_size", 2048)
  # Layer-related flags. If zero, these fall back on hparams.num_hidden_layers.
  hparams.add_hparam("num_encoder_layers", 0)
  hparams.add_hparam("num_decoder_layers", 0)
  # Attention-related flags.
  hparams.add_hparam("num_heads", 8)
  hparams.add_hparam("attention_key_channels", 0)
  hparams.add_hparam("attention_value_channels", 0)
  hparams.add_hparam("ffn_layer", "dense_relu_dense")
  hparams.add_hparam("parameter_attention_key_channels", 0)
  hparams.add_hparam("parameter_attention_value_channels", 0)
  # All hyperparameters ending in "dropout" are automatically set to 0.0
  # when not in training mode.
  hparams.add_hparam("attention_dropout", 0.0)
  hparams.add_hparam("attention_dropout_broadcast_dims", "")
  hparams.add_hparam("relu_dropout", 0.0)
  hparams.add_hparam("relu_dropout_broadcast_dims", "")
  hparams.add_hparam("pos", "timing")  # timing, none
  hparams.add_hparam("nbr_decoder_problems", 1)
  hparams.add_hparam("proximity_bias", False)
  hparams.add_hparam("causal_decoder_self_attention", True)
  hparams.add_hparam("use_pad_remover", True)
  hparams.add_hparam("self_attention_type", "dot_product")
  hparams.add_hparam("encdec_attention_type", "dot_product")
  hparams.add_hparam("dec_attention_type", "dot_product")
  hparams.add_hparam("conv_first_kernel", 3)
  hparams.add_hparam("attention_variables_3d", False)
  hparams.add_hparam("use_target_space_embedding", True)
  # These parameters are only used when ffn_layer=="local_moe_tpu"
  hparams.add_hparam("moe_overhead_train", 1.0)
  hparams.add_hparam("moe_overhead_eval", 2.0)
  hparams.moe_num_experts = 16
  hparams.moe_loss_coef = 1e-3
  # If specified, use this value instead of problem name in metrics.py.
  # This is useful for programs that can automatically compare experiments side
  #   by side based on the same metric names.
  hparams.add_hparam("overload_eval_metric_name", "")
  hparams.add_hparam("light_mode", "vanilla")
  hparams.add_hparam("mini_dim", 3)
  hparams.add_hparam("factor", 5)
  hparams.add_hparam("num_slots", 5)
  hparams.add_hparam("upsize", 1.0)
  hparams.add_hparam("ffn_upsize", 1)
  hparams.add_hparam('dual', 0)
  hparams.add_hparam('cut_length', 0)
  return hparams


@registry.register_hparams
def light_transformer_base_v2():
  """Set of hyperparameters."""
  hparams = light_transformer_base_v1()
  hparams.layer_preprocess_sequence = "n"
  hparams.layer_postprocess_sequence = "da"
  hparams.layer_prepostprocess_dropout = 0.1
  hparams.attention_dropout = 0.1
  hparams.relu_dropout = 0.1
  hparams.learning_rate_warmup_steps = 8000
  hparams.learning_rate = 0.2
  return hparams

@registry.register_hparams
def light_transformer_base_v3():
  """Base parameters for Transformer model."""
  # Update parameters here, then occasionally cut a versioned set, e.g.
  # transformer_base_v2.
  hparams = light_transformer_base_v2()
  hparams.optimizer_adam_beta2 = 0.997
  # New way of specifying learning rate schedule.
  # Equivalent to previous version.
  hparams.learning_rate_schedule = (
      "constant*linear_warmup*rsqrt_decay*rsqrt_hidden_size")
  hparams.learning_rate_constant = 2.0
  return hparams


@registry.register_hparams
def light_transformer_base():
  """Base parameters for Transformer model."""
  hparams = light_transformer_base_v3()
  return hparams

@registry.register_hparams
def light_transformer_base_mem8():
  """Base parameters for Transformer model."""
  hparams = light_transformer_base_v3()
  hparams.num_mem = 8
  return hparams

@registry.register_hparams
def light_transformer_base_mem16():
  """Base parameters for Transformer model."""
  hparams = light_transformer_base_v3()
  hparams.num_mem = 16
  return hparams



@registry.register_hparams
def light_transformer_big():
  """HParams for transformer big model on WMT."""
  hparams = light_transformer_base()
  hparams.hidden_size = 1024
  hparams.filter_size = 4096
  # Reduce batch size to 2048 from 4096 to be able to train the model on a GPU
  # with 12 GB memory. For example, NVIDIA TITAN V GPU.
  hparams.batch_size = 2048
  hparams.num_heads = 16
  hparams.layer_prepostprocess_dropout = 0.3
  return hparams


@registry.register_hparams
def light_transformer_tall():
  """Hparams for transformer on LM for pretraining/finetuning/mixing."""
  hparams = transformer_base()
  hparams.batch_size = 2048
  hparams.hidden_size = 768
  hparams.filter_size = 3072
  hparams.num_hidden_layers = 12
  hparams.num_heads = 12
  hparams.label_smoothing = 0.0
  hparams.max_length = 512
  hparams.eval_drop_long_sequences = True
  hparams.multiproblem_mixing_schedule = "pretrain"
  return hparams


@registry.register_hparams
def light_transformer_tall_finetune_textclass():
  """Hparams for transformer on LM for finetuning on text class problems."""
  hparams = transformer_tall()
  hparams.learning_rate_constant = 6.25e-5
  hparams.learning_rate_schedule = (
      "linear_warmup*constant*linear_decay")
  hparams.multiproblem_schedule_max_examples = 0
  hparams.multiproblem_target_eval_only = True
  hparams.multiproblem_class_loss_multiplier = 4
  hparams.learning_rate_warmup_steps = 50
  # Set train steps to learning_rate_decay_steps or less
  hparams.learning_rate_decay_steps = 25000
  hparams.multiproblem_reweight_label_loss = True
  hparams.multiproblem_label_weight = 0.95
  return hparams


@registry.register_hparams
def light_transformer_tall_pretrain_lm():
  """Hparams for transformer on LM pretraining (with 64k vocab)."""
  hparams = transformer_tall()
  hparams.learning_rate_constant = 2e-4
  hparams.learning_rate_schedule = (
      "linear_warmup*constant*cosdecay")
  hparams.optimizer = "AdamW"
  hparams.optimizer_adam_beta1 = 0.9
  hparams.optimizer_adam_beta2 = 0.999
  hparams.optimizer_adam_epsilon = 1e-8
  # Set max examples to something big when pretraining only the LM, definitely
  # something an order of magnitude bigger than number of train steps.
  hparams.multiproblem_schedule_max_examples = 5e8
  # Set train steps to learning_rate_decay_steps or less
  hparams.learning_rate_decay_steps = 5000000
  return hparams


@registry.register_hparams
def light_transformer_tall_pretrain_lm_tpu_adafactor():
  """Hparams for transformer on LM pretraining (with 64k vocab) on TPU."""
  hparams = transformer_tall_pretrain_lm()
  update_hparams_for_tpu(hparams)
  hparams.max_length = 1024
  # For multi-problem on TPU we need it in absolute examples.
  hparams.batch_size = 8
  hparams.multiproblem_vocab_size = 2**16
  return hparams


@registry.register_hparams
def light_transformer_tall_pretrain_lm_tpu_adafactor_large():
  """Hparams for transformer on LM pretraining on TPU, large model."""
  hparams = transformer_tall_pretrain_lm_tpu_adafactor()
  hparams.hidden_size = 1024
  hparams.num_heads = 16
  hparams.filter_size = 32768
  hparams.batch_size = 4
  hparams.multiproblem_mixing_schedule = "constant"
  hparams.multiproblem_schedule_threshold = 0.3
  return hparams


@registry.register_hparams
def light_transformer_tall_pretrain_lm_tpu():
  """Hparams for transformer on LM pretraining on TPU with AdamW."""
  hparams = transformer_tall_pretrain_lm_tpu_adafactor()
  # Optimizer gets reset in update_hparams_for_tpu so we set it again here.
  hparams.learning_rate_constant = 2e-4
  hparams.learning_rate_schedule = ("linear_warmup * constant * cosdecay")
  hparams.optimizer = "AdamW"
  return hparams


@registry.register_hparams
def light_transformer_tall_finetune_cnndm():
  """Hparams for transformer on LM for finetuning on cnndm summarization."""
  hparams = transformer_tall()
  hparams.batch_size = 4096
  hparams.multiproblem_max_input_length = 412
  hparams.multiproblem_max_target_length = 100
  hparams.multiproblem_schedule_max_examples = 0
  hparams.learning_rate_schedule = (
      "linear_warmup*constant*cosdecay")
  hparams.learning_rate_constant = 5e-5
  hparams.learning_rate_warmup_steps = 100
  # Set train steps to learning_rate_decay_steps or less
  hparams.learning_rate_decay_steps = 40000
  hparams.multiproblem_target_eval_only = True
  hparams.multiproblem_vocab_size = 2**16
  return hparams


@registry.register_hparams
def light_transformer_tall_big():
  """Hparams for transformer on LM+MNLI."""
  hparams = transformer_tall()
  hparams.num_hidden_layers = 18
  return hparams


@registry.register_hparams
def light_transformer_big_single_gpu():
  """HParams for transformer big model for single GPU."""
  hparams = light_transformer_big()
  hparams.layer_prepostprocess_dropout = 0.1
  hparams.learning_rate_warmup_steps = 16000
  return hparams


@registry.register_hparams
def light_transformer_base_single_gpu():
  """HParams for transformer base model for single GPU."""
  hparams = light_transformer_base()
  hparams.batch_size = 2048
  hparams.learning_rate_warmup_steps = 16000
  hparams.mem_size = 4
  return hparams

@registry.register_hparams
def light_transformer_base_single_gpu_mem8():
  """HParams for transformer base model for single GPU."""
  hparams = light_transformer_base()
  hparams.batch_size = 2048
  hparams.learning_rate_warmup_steps = 16000
  hparams.num_mem = 8
  return hparams

@registry.register_hparams
def light_transformer_base_multistep8():
  """HParams for simulating 8 GPUs with MultistepAdam optimizer."""
  hparams = light_transformer_base()
  hparams.optimizer = "MultistepAdam"
  hparams.optimizer_multistep_accumulate_steps = 8
  return hparams


@registry.register_hparams
def light_transformer_parsing_base():
  """HParams for parsing on WSJ only."""
  hparams = transformer_base()
  hparams.attention_dropout = 0.2
  hparams.layer_prepostprocess_dropout = 0.2
  hparams.max_length = 512
  hparams.learning_rate_warmup_steps = 16000
  hparams.hidden_size = 1024
  hparams.learning_rate = 0.05
  hparams.shared_embedding_and_softmax_weights = False
  return hparams


@registry.register_hparams
def light_transformer_parsing_big():
  """HParams for parsing on WSJ semi-supervised."""
  hparams = transformer_big()
  hparams.max_length = 512
  hparams.shared_source_target_embedding = False
  hparams.learning_rate_warmup_steps = 4000
  hparams.layer_prepostprocess_dropout = 0.1
  hparams.batch_size = 2048
  hparams.learning_rate = 0.05
  return hparams


@registry.register_hparams
def light_transformer_parsing_ice():
  """HParams for parsing and tagging Icelandic text."""
  hparams = transformer_base_single_gpu()
  hparams.batch_size = 4096
  hparams.shared_embedding_and_softmax_weights = False
  return hparams


@registry.register_hparams
def light_transformer_tiny():
  hparams = light_transformer_base()
  hparams.num_hidden_layers = 2
  hparams.hidden_size = 128
  hparams.filter_size = 512
  hparams.num_heads = 4
  return hparams


@registry.register_hparams
def light_transformer_test():
  hparams = transformer_base()
  hparams.num_hidden_layers = 2
  hparams.hidden_size = 16
  hparams.filter_size = 8
  hparams.num_heads = 2
  return hparams


@registry.register_hparams
def light_transformer_small():
  hparams = light_transformer_base()
  hparams.num_hidden_layers = 2
  hparams.hidden_size = 256
  hparams.filter_size = 1024
  hparams.num_heads = 4
  return hparams


@registry.register_hparams
def light_transformer_l2():
  hparams = transformer_base()
  hparams.num_hidden_layers = 2
  return hparams


@registry.register_hparams
def light_transformer_l4():
  hparams = transformer_base()
  hparams.num_hidden_layers = 4
  return hparams


@registry.register_hparams
def light_transformer_l8():
  hparams = transformer_base()
  hparams.num_hidden_layers = 8
  return hparams


@registry.register_hparams
def light_transformer_l10():
  hparams = transformer_base()
  hparams.num_hidden_layers = 10
  return hparams


@registry.register_hparams
def light_transformer_h1():
  hparams = transformer_base()
  hparams.num_heads = 1
  return hparams


@registry.register_hparams
def light_transformer_prepend_v2():
  hparams = light_transformer_base_v2()
  hparams.prepend_mode = "prepend_inputs_masked_attention"
  hparams.max_length = 0
  return hparams


@registry.register_hparams
def light_transformer_prepend_v1():
  hparams = light_transformer_base_v1()
  hparams.prepend_mode = "prepend_inputs_masked_attention"
  hparams.max_length = 0
  return hparams


@registry.register_hparams
def light_transformer_prepend():
  return light_transformer_prepend_v2()

@registry.register_hparams
def light_transformer_prepend_mem_8():
  hparams.num_mem=8
  return light_transformer_prepend_v2()

@registry.register_hparams
def light_transformer_relative():
  """Use relative position embeddings instead of absolute position encodings."""
  hparams = light_transformer_base()
  hparams.pos = None
  hparams.self_attention_type = "dot_product_relative"
  hparams.max_relative_position = 20
  return hparams


@registry.register_hparams
def light_transformer_relative_tiny():
  hparams = light_transformer_relative()
  hparams.num_hidden_layers = 2
  hparams.hidden_size = 128
  hparams.filter_size = 512
  hparams.num_heads = 4
  return hparams


@registry.register_hparams
def light_transformer_relative_big():
  hparams = light_transformer_big()
  hparams.pos = None
  hparams.self_attention_type = "dot_product_relative"
  hparams.max_relative_position = 20
  return hparams


@registry.register_hparams
def light_transformer_timeseries():
  hparams = light_transformer_small()
  hparams.batch_size = 256
  hparams.learning_rate_warmup_steps = 2000
  return hparams


