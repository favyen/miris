import model as model

import json
import numpy
import os
import math
import random
import subprocess
import sys
import tensorflow as tf
import time

VIDEO = 'shibuya'
SUFFIX = '-shibuya'
SKIP = 24
#PATH = '/mnt/dji/miris-models/{}/filter-rnn-shibuyabt{}/'.format(VIDEO, SKIP)
#PATH = '/mnt/dji/miris-models/{}/rnn-suffix{}/'.format(VIDEO, SKIP)
PATH = '/mnt/dji/miris-models/{}/filter-rnn{}/'.format(VIDEO, SKIP)
JSON_PATHS = [
	'/mnt/dji/miris-models/{}/filter-rnn{}/train_true.json'.format(VIDEO, SUFFIX),
	'/mnt/dji/miris-models/{}/filter-rnn{}/train_false.json'.format(VIDEO, SUFFIX),
	'/mnt/dji/miris-models/{}/filter-rnn{}/val_true.json'.format(VIDEO, SUFFIX),
	'/mnt/dji/miris-models/{}/filter-rnn{}/val_false.json'.format(VIDEO, SUFFIX),
	#'/mnt/dji/miris-models/warsaw/count-ps12.json',
]
NORM_SIZE = 1000.0

def load(json_path, label):
	with open(json_path, 'r') as f:
		tracks = json.load(f)
	return [(track, label) for track in tracks]

def load2(json_path, sel):
	with open(json_path, 'r') as f:
		stuff = json.load(f)
	out = []
	for d in stuff:
		out.append((d['track'], d[sel]))
	return out

if True:
	COARSE_INPUT = False
	train_pairs = load(JSON_PATHS[0], 1) + load(JSON_PATHS[1], 0)
	val_pairs = load(JSON_PATHS[2], 1) + load(JSON_PATHS[3], 0)

	random.shuffle(train_pairs)
	random.shuffle(val_pairs)
	train_good_pairs = [pair for pair in train_pairs if pair[1] == 1]
	val_good_pairs = [pair for pair in val_pairs if pair[1] == 1]
else:
	COARSE_INPUT = True
	all_pairs = load2(JSON_PATHS[0], 'need_suffix')
	random.shuffle(all_pairs)
	num_val = len(all_pairs)/5+1
	train_pairs = all_pairs[num_val:]
	val_pairs = all_pairs[0:num_val]

def get_track_data(track):
	data = []
	for detection in track:
		cx = (detection['left']+detection['right'])/2/NORM_SIZE
		cy = (detection['top']+detection['bottom'])/2/NORM_SIZE
		width = (detection['right'] - detection['left'])/NORM_SIZE
		height = (detection['bottom']-detection['top'])/NORM_SIZE
		data.append([cx, cy, width, height])
	return data

def extract(pair, skip=SKIP):
	track, label = pair
	if COARSE_INPUT:
		start_idx = track[0]['frame_idx'] % skip
	else:
		start_idx = random.randint(0, skip-1)
	coarse = [d for d in track if d['frame_idx'] % skip == start_idx]
	coarse = get_track_data(coarse)
	if len(coarse) == 0:
		return extract(pair)
	l = len(coarse)
	if l > model.MAX_LENGTH:
		return extract(pair, skip=skip*2)
	while len(coarse) < model.MAX_LENGTH:
		coarse.append([0, 0, 0, 0])
	return coarse, l, label

#val_examples = [extract(pair) for pair in val_pairs]
val_examples = [extract(pair) for pair in val_good_pairs] + [extract(random.choice(val_pairs)) for _ in xrange(len(val_good_pairs))]

train_examples = [extract(random.choice(train_pairs)) for _ in xrange(4*len(train_pairs))] + [extract(random.choice(train_good_pairs)) for _ in xrange(4*len(train_pairs))]

print 'initializing model'
m = model.Model()
session = tf.Session()
session.run(m.init_op)
latest_path = '{}/model_latest/model'.format(PATH)
best_path = '{}/model_best/model'.format(PATH)

print 'begin training'
best_loss = None

for epoch in xrange(9999):
	start_time = time.time()
	train_losses = []
	for _ in xrange(128):
		#batch_examples = random.sample(train_pairs, model.BATCH_SIZE) + random.sample(train_good_pairs, model.BATCH_SIZE)
		#batch_examples = [extract(pair) for pair in batch_examples]
		batch_examples = random.sample(train_examples, model.BATCH_SIZE)
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

	m.saver.save(session, latest_path)
	if best_loss is None or val_loss < best_loss:
		best_loss = val_loss
		m.saver.save(session, best_path)

def get_stats():
	def score(pairs, threshold=None):
		true_probs = []
		false_probs = []
		for pair in pairs:
			example = extract(pair)
			output = session.run(m.outputs, feed_dict={
				m.is_training: False,
				m.inputs: [example[0]],
				m.lengths: [example[1]],
			})[0]
			if pair[1] == 0:
				false_probs.append(output)
			else:
				true_probs.append(output)
		print '... true: {} +/- {} ({} to {})'.format(numpy.mean(true_probs), numpy.std(true_probs), numpy.min(true_probs), numpy.max(true_probs))
		print '... false: {} +/- {} ({} to {})'.format(numpy.mean(false_probs), numpy.std(false_probs), numpy.min(false_probs), numpy.max(false_probs))

		if threshold is None:
			return true_probs, false_probs, numpy.min(true_probs)

		match = 0
		fp = 0
		fn = 0
		for prob in true_probs:
			if prob >= threshold:
				match += 1
			else:
				fn += 1
		for prob in false_probs:
			if prob >= threshold:
				fp += 1
		print '... stats: match={}, fp={}, fn={} ... precision={} recall={}'.format(match, fp, fn, float(match)/float(match+fp), float(match)/float(match+fn))
		return true_probs, false_probs, numpy.min(true_probs)

	print 'train:'
	train_true_probs, train_false_probs, threshold = score(train_pairs)
	print 'val:'
	val_true_probs, val_false_probs, _ = score(val_pairs, threshold=threshold)

def apply(threshold, in_path, out_path):
	def get_tracks(detections):
		track_dict = {}
		for frame_idx in xrange(len(detections)):
			if detections[frame_idx] is None:
				continue
			for detection in detections[frame_idx]:
				track_id = detection['track_id']
				if track_id not in track_dict:
					track_dict[track_id] = []
				track_dict[track_id].append(detection)
		return track_dict.values()

	def tracks_to_detections(tracks):
		detections = []
		for track in tracks:
			for detection in track:
				while detection['frame_idx'] >= len(detections):
					detections.append([])
				detections[detection['frame_idx']].append(detection)
		return detections

	with open(in_path, 'r') as f:
		detections = json.load(f)
	tracks = get_tracks(detections)
	good_tracks = []
	for track in tracks:
		track_data = get_track_data(track)
		while len(track_data) > model.MAX_LENGTH:
			track_data = [t for i, t in enumerate(track_data) if i % 2 == 0]
		l = len(track_data)
		while len(track_data) < model.MAX_LENGTH:
			track_data.append([0, 0, 0, 0])
		output = session.run(m.outputs, feed_dict={
			m.is_training: False,
			m.inputs: [track_data],
			m.lengths: [l],
		})[0]
		if output < threshold:
			continue
		good_tracks.append(track)
	print 'filter from {} to {}'.format(len(tracks), len(good_tracks))

	detections = tracks_to_detections(good_tracks)
	with open(out_path, 'w') as f:
		json.dump(detections, f)
