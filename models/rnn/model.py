import numpy
import tensorflow as tf
import os
import os.path
import random
import math
import time
from PIL import Image

BATCH_SIZE = 4
MAX_LENGTH = 64

class Model:
	def _fc_layer(self, name, input_var, input_size, output_size, options = {}):
		activation = options.get('activation', 'relu')
		dropout = options.get('dropout', None)
		batchnorm = options.get('batchnorm', False)

		with tf.variable_scope(name) as scope:
			weights = tf.get_variable(
				'weights',
				shape=[input_size, output_size],
				initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0 / input_size)),
				dtype=tf.float32
			)
			biases = tf.get_variable(
				'biases',
				shape=[output_size],
				initializer=tf.constant_initializer(0.0),
				dtype=tf.float32
			)
			output = tf.matmul(input_var, weights) + biases
			if batchnorm:
				output = tf.contrib.layers.batch_norm(output, center=True, scale=True, is_training=self.is_training, decay=0.99)
			if dropout is not None:
				output = tf.nn.dropout(output, keep_prob=1-dropout)

			if activation == 'relu':
				return tf.nn.relu(output, name=scope.name)
			elif activation == 'sigmoid':
				return tf.nn.sigmoid(output, name=scope.name)
			elif activation == 'none':
				return output
			else:
				raise Exception('invalid activation {} specified'.format(activation))

	def __init__(self):
		tf.reset_default_graph()

		self.is_training = tf.placeholder(tf.bool)
		self.inputs = tf.placeholder(tf.float32, [None, MAX_LENGTH, 4])
		self.targets = tf.placeholder(tf.float32, [None])
		self.lengths = tf.placeholder(tf.int32, [None])
		self.learning_rate = tf.placeholder(tf.float32)

		self.layer0 = tf.reshape(
			self._fc_layer('layer0', tf.reshape(self.inputs, [-1, 4]), 4, 32),
			[-1, MAX_LENGTH, 32]
		)
		with tf.variable_scope('rnn_cell') as scope:
			self.rnn_cell = tf.nn.rnn_cell.BasicRNNCell(32)
			_, self.rnn_outputs = tf.nn.dynamic_rnn(
				cell=self.rnn_cell,
				inputs=self.layer0,
				sequence_length=self.lengths,
				dtype=tf.float32
			)
		self.pre_outputs = self._fc_layer('pre_outputs', self.rnn_outputs, 32, 1, {'activation': 'none'})[:, 0]
		self.outputs = tf.nn.sigmoid(self.pre_outputs)
		self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.targets, logits=self.pre_outputs))

		with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
			self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

		self.init_op = tf.initialize_all_variables()
		self.saver = tf.train.Saver(max_to_keep=None)
