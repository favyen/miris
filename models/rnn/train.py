import model as model

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import json
import numpy
import math
import random
import subprocess
import sys
import tensorflow as tf
import time

data_path = sys.argv[1]
model_path = sys.argv[2]

with open(data_path, 'r') as f:
	data = json.load(f)

def load_func(l):
	pairs = []
	for d in l:
		seq, length = model.pad_track(d['track'])
		pairs.append((seq, length, d['label']))
	return pairs

train_examples = load_func(data['train'])
val_examples = load_func(data['val'])
num_outputs = len(train_examples[0][2])

train_true = [t for t in train_examples if any(t[2])]
train_false = [t for t in train_examples if not any(t[2])]
def sample_func():
	return random.sample(train_true, model.BATCH_SIZE/2) + random.sample(train_false, model.BATCH_SIZE/2)

val_true = [t for t in val_examples if any(t[2])]
val_false = [t for t in val_examples if not any(t[2])]
num_sel = min(128, len(val_true))
val_examples = random.sample(val_true, min(num_sel, len(val_true))) + random.sample(val_false, min(num_sel, len(val_false)))
random.shuffle(val_examples)

print 'initializing model'
m = model.Model(num_outputs=num_outputs)
config = tf.ConfigProto(
	device_count={'GPU': 0}
)
session = tf.Session(config=config)
session.run(m.init_op)

print 'begin training'
best_loss = None

for epoch in xrange(9999):
	start_time = time.time()
	train_losses = []
	for _ in xrange(128):
		batch_examples = sample_func()
		_, loss = session.run([m.optimizer, m.loss], feed_dict={
			m.is_training: True,
			m.inputs: [example[0] for example in batch_examples],
			m.targets: [example[2] for example in batch_examples],
			m.lengths: [example[1] for example in batch_examples],
			m.learning_rate: 1e-3,
		})
		train_losses.append(loss)
	train_loss = numpy.mean(train_losses)
	train_time = time.time()

	val_losses = []
	for i in xrange(0, len(val_examples), model.BATCH_SIZE):
		batch_examples = val_examples[i:i+model.BATCH_SIZE]
		loss, outputs = session.run([m.loss, m.outputs], feed_dict={
			m.is_training: False,
			m.inputs: [example[0] for example in batch_examples],
			m.targets: [example[2] for example in batch_examples],
			m.lengths: [example[1] for example in batch_examples],
		})
		val_losses.append(loss)

	val_loss = numpy.mean(val_losses)
	val_time = time.time()

	print 'iteration {}: train_time={}, val_time={}, train_loss={}, val_loss={}/{}'.format(epoch, int(train_time - start_time), int(val_time - train_time), train_loss, val_loss, best_loss)

	if best_loss is None or val_loss < best_loss:
		best_loss = val_loss
		m.saver.save(session, model_path)
