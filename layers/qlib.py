from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from numpy.random import RandomState
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import random_ops
import random

""" Quarternion layers

References:

https://arxiv.org/pdf/1806.04418.pdf
https://arxiv.org/pdf/1806.07789.pdf

https://github.com/Orkis-Research/light-Recurrent-Neural-Networks
https://github.com/Orkis-Research/light-Convolutional-Neural-Networks-for-End-to-End-Automatic-Speech-Recognition

Some functions are direct ports from the Pytorch library.

"""

def make_quarternion_mul(kernel, concat_dim=0):
	r, i, j, k = tf.split(kernel, 4, axis=-1)
	r2 = tf.concat([r, -i, -j, -k], axis=-1)	# 0, 1, 2, 3
	i2 = tf.concat([i, r, -k, j], axis=-1)	# 1, 0, 3, 2
	j2 = tf.concat([j, k, r, -i], axis=-1)	# 2, 3, 0, 1
	k2 = tf.concat([k, -j, i, r],axis=-1)	# 3, 2, 1, 0
	hamilton = tf.concat([r2, i2, j2, k2], axis=concat_dim)
	return hamilton


def get_r(x, a=1):
	return tf.split(x, 4, axis=a)[0]

def get_i(x, a=1):
	return tf.split(x, 4, axis=a)[1]

def get_j(x, a=1):
	return tf.split(x, 4, axis=a)[2]

def get_k(x, a=1):
	return tf.split(x, 4, axis=a)[3]

def quarternion_attention(a, b):
	""" Performs dot product attention between two quarternion sequences.

	a = bsz x al x dim
	b = bsz x bl x dim

	following:
	(rr' - xx' - yy' - zz')  +
		(rx' + xr' + yz' - zy')i +
		(ry' - xz' + yr' + zx')j +
		(rz' + xy' - yx' + zr')k +

	the output should be one attention matrix for each component (r,i,j,k)
	"""
	print("light Attention!")
	print(a)
	print(b)
	al, bl = tf.shape(a)[2], tf.shape(b)[2]

	ar, ax, ay, az = tf.split(a, 4, axis=-1)
	br, bx, by, bz = tf.split(b, 4, axis=-1)
	r = tf.matmul(ar, br, transpose_b=True) - tf.matmul(ax, bx, transpose_b=True) - tf.matmul(ay, by, transpose_b=True) - tf.matmul(az, bz, transpose_b=True)
	i = tf.matmul(ar, bx, transpose_b=True) + tf.matmul(ax, br, transpose_b=True) + tf.matmul(ay, bz, transpose_b=True) - tf.matmul(az, by, transpose_b=True)
	j = tf.matmul(ar, by, transpose_b=True) - tf.matmul(ax, bz, transpose_b=True) + tf.matmul(ay, br, transpose_b=True) + tf.matmul(az, bx, transpose_b=True)
	k = tf.matmul(ar, bz, transpose_b=True) + tf.matmul(ax, by, transpose_b=True) - tf.matmul(ay, bx, transpose_b=True) + tf.matmul(az, br, transpose_b=True)
	return [r, i, j, k]

def quarternion_dot_product_att(a, b):
	""" Wrapper for two sequences
	"""
	al = tf.shape(a)[1]
	bl = tf.shape(b)[1]
	# print(a)
	d = a.get_shape().as_list()[2]
	bsz = tf.shape(b)[0]
	a = tf.reshape(a, [-1, d])
	a = tf.tile(a, [bl, 1])
	b = tf.reshape(b, [-1, d])
	b = tf.tile(b, [al, 1])
	att = quarternion_dot(a, b)
	att = tf.reshape(att, [bsz, -1, al * bl])
	att = tf.reduce_sum(att, 1)
	return tf.reshape(att, [-1, al * bl])

def quarternion_dot_3d(q0, q1):
	d = q0.get_shape().as_list()[2]
	sq = tf.shape(q0)[1]
	q0 = tf.reshape(q0, [-1, d])
	q1 = tf.reshape(q1, [-1, d])
	out = quarternion_dot(q0, q1)
	return tf.reshape(out, [-1, sq, d])

def quarternion_dot(q0, q1):
	""" Quarternion product between 2 quarternions

	returns same shape and acts like element-wise quarternion mul
	"""
	q1_r = get_r(q1)
	q1_i = get_i(q1)
	q1_j = get_j(q1)
	q1_k = get_k(q1)

	r_base = tf.multiply(q0, q1)
	r = get_r(r_base) - get_i(r_base) - get_j(r_base) - get_k(r_base)

	i_base = tf.multiply(q0, tf.concat([q1_i, q1_r, q1_k, q1_j], 1))
	i = get_r(i_base) + get_i(i_base) + get_j(i_base) - get_k(i_base)

	j_base = tf.multiply(q0, tf.concat([q1_j, q1_k, q1_r, q1_i], 1))
	j = get_r(j_base) - get_i(j_base) + get_j(j_base) + get_k(j_base)

	k_base = tf.multiply(q0, tf.concat([q1_k, q1_j, q1_i, q1_r], 1))
	k = get_r(k_base) + get_i(k_base) - get_j(k_base) + get_k(k_base)

	return tf.concat([r, i, j, k], 1)

def quarternion_concat(x, axis):
	""" Helpful if we have 2 quarternions in [r,i,j,k].
	We can't simply concat them as it would mess the components.
	So in this case, we extract each component and concat them individually.
	"""
	output = [[] for i in range(4)]
	for _x in x:
		sp = tf.split(_x, 4, axis=axis)
		for i in range(4):
			output[i].append(sp[i])

	final = []
	for o in output:
		o = tf.concat(o, axis)
		final.append(o)

	return tf.concat(final, axis)

def quarternion_ffn_3d(x, dim, name='', init=None,
				num_layers=1, activation=None, reuse=None):
	""" Quarternion Feed-forward layers to 3D input [bsz x seq_len x dim]
	returns same shape tensor with new projected dimension.
	"""
	print("QFFN layer..")
	_d = x.get_shape().as_list()[2]
	sq = tf.shape(x)[1]
	x = tf.reshape(x, [-1, _d])
	x = quarternion_ffn(x, dim, name=name, init=init,
						num_layers=num_layers,
						activation=activation,reuse=reuse)
	x = tf.reshape(x, [-1, sq, dim])
	return x

def factorized_ffn_3d(x, dim, name='', init=None,
				num_layers=1, activation=None, reuse=None):
	""" 3D factorized FFN layer
	"""
	print("Factor Layer")
	_d = x.get_shape().as_list()[2]
	sq = tf.shape(x)[1]
	x = tf.reshape(x, [-1, _d])
	x = factorized_ffn(x, dim, name=name, init=init,
						num_layers=num_layers,
						activation=activation,reuse=reuse)
	x = tf.reshape(x, [-1, sq, dim])
	return x


def factorized_ffn(x, dim, name='', init=None,
				num_layers=1, activation=None, reuse=None):
	""" Factorized FFN
	"""
	if(init is None):
		init = tf.contrib.layers.xavier_initializer()
	input_dim=x.get_shape().as_list()[2]
	k1 = tf.get_variable('factork1{}'.format(name), [input_dim], initializer=init)
	k2 = tf.get_variable('factork2{}'.format(name), [dim], initializer=init)
	W = tf.tensordot(k1, k2, axes=0)
	output = tf.matmul(x, W)
	if(activation):
		output = activation(output)
	return output

def quarternion_ffn(x, dim, name='', init=None,
				num_layers=1, activation=None, reuse=None):
	""" Implements quarternion feed-forward layer

	x is [bsz x features] tensor
	"""
	if(init is None):
		init = tf.contrib.layers.xavier_initializer()
		# init = q_xavier_initializer()
	input_dim = x.get_shape().as_list()[1] // 4
	with tf.variable_scope('Q{}'.format(name), reuse=reuse) as scope:
		kernel = tf.get_variable('quarternion', [input_dim, dim], initializer=init)
		hamilton = make_quarternion_mul(kernel)
		output = tf.matmul(x, hamilton)
		if(activation):
			output = activation(output)
		return output

def make_random_mul(kernel, n=4, concat_dim=0, dual=False):
	""" input is dim/n x dim
	output is dim x dim

	generalization and parameterized hypercomplex product
	"""
	dim = kernel.get_shape().as_list()[1]
	dim2 = kernel.get_shape().as_list()[0]
	kernel = tf.reshape(kernel, [dim2, 1, 1, dim])
	mix = tf.split(kernel, n, axis=-1)
	sdim = mix[0].get_shape().as_list()[-1]	# dim//n x 1 x 1 x dim//n

	AM = tf.get_variable('A', [n, 1, n, n])

	cat = tf.concat(mix, axis=1) # dim/n x n x 1 x  dim/n
	cat = tf.tile(cat, [1, 1, n, 1])	# dim/n x n x n x dim/n
	cat = tf.transpose(cat, [1, 0, 2, 3])	# n x dim/n x n x dim/n

	if(dual==1):
		print("Using Dual..")
		BM = tf.get_variable('B', [n, 1, n, n])
		AM *= tf.nn.sigmoid(BM)

	AM = tf.tile(AM, [1, dim2, 1, 1])	# n x dim/n x n x n
	cat = tf.matmul(AM, cat)	# n x dim/n x n x dim/n
	output = tf.reshape(cat, [dim2 *n, dim])
	return output


def random_ffn_3d(x, dim, n=16, name='', init=None,
				num_layers=1, activation=None, reuse=None, dual=False):
	""" Implements random feed-forward layer

	x is [bsz x features] tensor
	"""
	print("R-FFN layer..n={} dual={}".format(n, dual))
	_d = x.get_shape().as_list()[2]
	sq = tf.shape(x)[1]
	x = tf.reshape(x, [-1, _d])
	print(x)
	x = random_ffn(x, dim, n=n, name=name, init=init,
						num_layers=num_layers,
						activation=activation, reuse=reuse, dual=dual)
	x = tf.reshape(x, [-1, sq, dim])
	return x


def random_ffn(x, dim, n=4, name='', init=None,
				num_layers=1, activation=None, reuse=None, dual=0):
	""" Implements random feed-forward layer

	x is [bsz x features] tensor
	"""
	if(init is None):
		init = tf.contrib.layers.xavier_initializer()
		# init = q_xavier_initializer()
	input_dim = x.get_shape().as_list()[1] // n
	with tf.variable_scope('R{}'.format(name), reuse=reuse) as scope:
		kernel = tf.get_variable('random', [input_dim, dim], initializer=init)
		hamilton = make_random_mul(kernel, n=n, dual=dual)
		output = tf.matmul(x, hamilton)
		if(activation):
			output = activation(output)
		return output


def octonion_ffn_3d(x, dim, name='', init=None,
				num_layers=1, activation=None, reuse=None):
	""" Quarternion Feed-forward layers to 3D input [bsz x seq_len x dim]
	returns same shape tensor with new projected dimension.
	"""
	print("OFFN layer..")
	_d = x.get_shape().as_list()[2]
	sq = tf.shape(x)[1]
	x = tf.reshape(x, [-1, _d])
	x = octonion_ffn(x, dim, name=name, init=init,
						num_layers=num_layers,
						activation=activation,reuse=reuse)
	x = tf.reshape(x, [-1, sq, dim])
	return x

def octonion_ffn(x, dim, name='', init=None,
				num_layers=1, activation=None, reuse=None):
	if(init is None):
		init = tf.contrib.layers.xavier_initializer()
	input_dim = x.get_shape().as_list()[1] // 8
	with tf.variable_scope('OCT{}'.format(name), reuse=reuse) as scope:
		kernel = tf.get_variable('octonion', [input_dim, dim], initializer=init)
		output = octonion_mul(x, kernel)
		return output


def hamilton_product(x, kernel):
	h = make_quarternion_mul(kernel)
	output = tf.matmul(x, h)
	return output

def qstar(x):
	x = tf.split(x, 4, axis=-1)
	x1 = -x[1]
	x2 = -x[2]
	return tf.concat([x[0],x1,x2,x[3]], axis=-1)

def octonion_mul(x, kernel):
	x1, x2 = tf.split(x, 2, axis=-1)
	k1, k2 = tf.split(kernel, 2, axis=-1)
	print(x1)
	print(k1)
	o1 = hamilton_product(x1, k1)
	o2 = hamilton_product(k2, x1)
	o1 -= hamilton_product(qstar(k2), x2)
	o2 += hamilton_product(x2, qstar(k1))
	output = tf.concat([o1, o2], axis=-1)
	return output

class QuarternionRNN(tf.nn.rnn_cell.RNNCell):

	def __init__(self, input_dim, output_dim,
					initializer=None, name='', reuse=None):
		""" Rough implementation (need double-check)
		from the Quarternion RNN paper. For now, works decently.
		"""
		self.dim = output_dim
		with tf.variable_scope("QuartRNN{}".format(name), reuse=reuse) as scope:
			if(initializer is None):
				# initializer = tf.contrib.layers.xavier_initializer()
				initialzier = tf.orthogonal_initializer()
			input_dim = input_dim // 4
			self.Wh = tf.get_variable("Wh", [input_dim, output_dim],
									initializer=initializer)
			self.Wx = tf.get_variable("Wx", [input_dim, output_dim],
									initializer=initializer)
			self.Wy = tf.get_variable("Wy", [input_dim, output_dim],
									initializer=initializer)
			self.Wh = make_quarternion_mul(self.Wh)
			self.Wx = make_quarternion_mul(self.Wx)
			self.Wy = make_quarternion_mul(self.Wy)

	@property
	def state_size(self):
		return self.dim

	@property
	def output_size(self):
		return self.dim


	def __call__(self, inputs, state, scope=None):
		"""
		inputs: 2-D tensor of shape [batch_size, feats + [gates]]
		"""
		new_state = tf.matmul(state, self.Wh) + tf.matmul(inputs, self.Wx)
		new_state = tf.nn.sigmoid(new_state)
		output = tf.nn.tanh(tf.matmul(inputs, self.Wy))
		return output, new_state
