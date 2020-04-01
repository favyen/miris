# like run4.py but we resize the bounding box to fixed size and add the x/y lengths in gnn

from discoverlib import geom
import model19 as model

import graph_nets
import json
import numpy
import os
from PIL import Image
import math
import random
import skimage.io, skimage.transform
import subprocess
import sys
import tensorflow as tf
import time

ORIG_WIDTH = 1920
ORIG_HEIGHT = 1080
PADDING = 0
PATH = '/mnt/dji/miris-models/mot/m19-skipvar/'
#FRAME_PATH = '/mnt/dji/drone-video/{}-half/'
#FRAME_PATH = '/mnt/bdd/bdd-selected-videos/frames/{}/'
#FRAME_PATH = '/data2/youtube/warsaw/frames-half/{}/'
FRAME_PATH = '/data2/mot/mot17/train/MOT17-{}-SDP/img1/'
DATA_PATHS = []
LOAD_SKIP = 1
SKIP = lambda: random.sample([1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128], 12)
FRAME_SCALE = 1
CROP_SIZE = 64
NUM_FEATURES = 64

#for label in ['0006', '0007', '0008', '0009', '0011']:
#	DATA_PATHS.append([
#		label,
#		'/mnt/dji/drone-video/json/{}-track-res960-freq1.json'.format(label),
#	])
#for fname in os.listdir('/mnt/bdd/bdd-selected-videos/videos/'):
#	if '.mov' not in fname:
#		continue
#	label = fname.split('.mov')[0]
#	DATA_PATHS.append([
#		label,
#		'/mnt/bdd/bdd-selected-videos/ped-json/{}-filter-track.json'.format(label),
#	])
#for label in ['0', '1', '3', '4', '5']:
#	DATA_PATHS.append([
#		label,
#		'/data2/youtube/warsaw/json/{}-track-res960-freq1.json'.format(label),
#	])
for label in ['02', '04', '05', '09', '10', '11', '13']:
	DATA_PATHS.append([
		label,
		'/data2/mot/mot17/train/MOT17-{}-SDP/gt/gt.json'.format(label),
	])

def get_frame_fname(frame_idx):
	s = str(frame_idx)
	while len(s) < 6:
		s = '0' + s
	return s + '.jpg'

def zip_frame_info(detections, label, frame_idx):
	if not detections:
		return []
	frame_path = FRAME_PATH.format(label)
	im = skimage.io.imread('{}/{}'.format(frame_path, get_frame_fname(frame_idx)))
	im_bounds = geom.Rectangle(
		geom.Point(0, 0),
		geom.Point(im.shape[0], im.shape[1])
	)
	info = []
	for idx, detection in enumerate(detections):
		rect = geom.Rectangle(
			geom.Point(detection['top']/FRAME_SCALE, detection['left']/FRAME_SCALE),
			geom.Point(detection['bottom']/FRAME_SCALE, detection['right']/FRAME_SCALE)
		)
		padding = rect.lengths().scale(PADDING)
		rect = geom.Rectangle(
			rect.start.sub(padding),
			rect.end.add(padding)
		)
		rect = im_bounds.clip_rect(rect)
		if rect.lengths().x < 4 or rect.lengths().y < 4:
			continue
		crop = im[rect.start.x:rect.end.x, rect.start.y:rect.end.y, :]
		resize_factor = min([float(CROP_SIZE) / crop.shape[0], float(CROP_SIZE) / crop.shape[1]])
		resize_shape = [int(crop.shape[0] * resize_factor), int(crop.shape[1] * resize_factor)]
		if resize_shape[0] == 0 or resize_shape[1] == 0:
			continue
		crop = (skimage.transform.resize(crop, resize_shape)*255).astype('uint8')
		fix_crop = numpy.zeros((CROP_SIZE, CROP_SIZE, 3), dtype='uint8')
		fix_crop[0:crop.shape[0], 0:crop.shape[1], :] = crop
		detection['width'] = float(detection['right']-detection['left'])/ORIG_WIDTH
		detection['height'] = float(detection['bottom']-detection['top'])/ORIG_HEIGHT
		info.append((detection, fix_crop, idx))
	return info

def get_loc(detection):
	cx = (detection['left'] + detection['right']) / 2
	cy = (detection['top'] + detection['bottom']) / 2
	cx = float(cx) / ORIG_WIDTH
	cy = float(cy) / ORIG_HEIGHT
	return cx, cy

def get_frame_pair(info1, info2, skip):
	# we need to get two graphs:
	# * input: (pos, features) for each node, (delta, distance, reverse) for each edge
	# * target: (match) for each edge
	# node <len(info1)> is special, it's for detections in cur frame that don't match any in next frame

	senders = []
	receivers = []
	input_nodes = []
	input_edges = []
	target_edges = []
	input_crops = []

	for i, t in enumerate(info1):
		detection, crop, _ = t
		cx, cy = get_loc(detection)
		input_nodes.append([cx, cy, detection['width'], detection['height'], 1, 0, 0, skip/50.0])# + [0.0]*NUM_FEATURES)
		input_crops.append(crop)
	input_nodes.append([0.5, 0.5, 0, 0, 0, 1, 0, skip/50.0])# + [0.0]*NUM_FEATURES)
	input_crops.append(numpy.zeros((CROP_SIZE, CROP_SIZE, 3), dtype='uint8'))
	for i, t in enumerate(info2):
		detection, crop, _ = t
		cx, cy = get_loc(detection)
		input_nodes.append([cx, cy, detection['width'], detection['height'], 0, 0, 1, skip/50.0])# + [0.0]*NUM_FEATURES)
		input_crops.append(crop)

	num_matches = 0
	for i, t1 in enumerate(info1):
		detection1, _, _ = t1
		x1, y1 = get_loc(detection1)
		does_match = False

		for j, t2 in enumerate(info2):
			detection2, _, _ = t2
			x2, y2 = get_loc(detection2)

			senders.extend([i, len(info1) + 1 + j])
			receivers.extend([len(info1) + 1 + j, i])
			edge_shared = [x2 - x1, y2 - y1, math.sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1))]
			input_edges.append(edge_shared + [1, 0, 0])
			input_edges.append(edge_shared + [0, 1, 0])

			if detection1['track_id'] == detection2['track_id']:
				label = 1.0
				does_match = True
			else:
				label = 0.0
			target_edges.append([label])
			target_edges.append([0.0])

		senders.extend([i, len(info1)])
		receivers.extend([len(info1), i])
		edge_shared = [0.0, 0.0, 0.0]
		input_edges.append(edge_shared + [0, 0, 0])
		input_edges.append(edge_shared + [1, 0, 0])

		if does_match:
			label = 0.0
			num_matches += 1
		else:
			label = 1.0
		target_edges.append([label])
		target_edges.append([0.0])

	if num_matches == 0 and False:
		return None, None, None

	def add_internal_edges(info, offset):
		for i, t1 in enumerate(info):
			detection1, _, _ = t1
			x1, y1 = get_loc(detection1)

			for j, t2 in enumerate(info):
				if i == j:
					continue

				detection2, _, _ = t2
				x2, y2 = get_loc(detection2)

				senders.append(offset + i)
				receivers.append(offset + j)
				input_edges.append([x2 - x1, y2 - y1, math.sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1))] + [0, 0, 1])
				target_edges.append([0.0])
	add_internal_edges(info1, 0)
	add_internal_edges(info2, len(info1) + 1)

	senders = numpy.array(senders, dtype='int32')
	receivers = numpy.array(receivers, dtype='int32')
	input_nodes = numpy.array(input_nodes, dtype='float32')
	input_edges = numpy.array(input_edges, dtype='float32')
	target_edges = numpy.array(target_edges, dtype='float32')

	input_dict = {
		"globals": [],
		"nodes": input_nodes,
		"edges": input_edges,
		"senders": senders,
		"receivers": receivers,
	}
	target_dict = {
		"globals": [],
		"nodes": numpy.zeros((len(input_nodes), 0), dtype='float32'),
		"edges": target_edges,
		"senders": senders,
		"receivers": receivers,
	}
	return input_dict, target_dict, input_crops, num_matches

all_pairs = [] # pairs of frames that are 8 apart
for label, detection_path in DATA_PATHS:
	print 'reading from {}'.format(detection_path)
	with open(detection_path, 'r') as f:
		detections = json.load(f)

	if not detections:
		continue

	frame_infos = {}
	def get_frame_info(frame_idx):
		if frame_idx not in frame_infos:
			frame_infos[frame_idx] = zip_frame_info(detections[frame_idx], label, frame_idx)
		return frame_infos[frame_idx]
	for frame_idx in xrange(0, len(detections), LOAD_SKIP):
		for skip in SKIP():
			if frame_idx+skip >= len(detections) or not detections[frame_idx] or not detections[frame_idx+skip]:
				continue
			print label, frame_idx
			info1 = get_frame_info(frame_idx)
			info2 = get_frame_info(frame_idx+skip)
			if len(info1) == 0 or len(info2) == 0:
				continue
			input_dict, target_dict, input_crops, num_matches = get_frame_pair(info1, info2, skip)
			if input_dict is None or target_dict is None:
				continue
			all_pairs.append((
				input_dict, target_dict, input_crops,
				label, frame_idx,
				[get_loc(detection) for detection, _, _ in info1],
				[get_loc(detection) for detection, _, _ in info2],
				num_matches
			))

random.shuffle(all_pairs)
num_val = min(1024, len(all_pairs)/10)
#val_pairs = all_pairs[:num_val]
#train_pairs = all_pairs[num_val:]

#all_pairs = [pair for pair in all_pairs if pair[3] != '0011' or pair[4] < 3200]
#val_pairs = [pair for pair in all_pairs if pair[3] == '0008' and pair[4] > 3200][0:1024]
#train_pairs = [pair for pair in all_pairs if pair[3] != '0008' or pair[4] <= 3200]

#all_pairs = [pair for pair in all_pairs if pair[3][0] in '0123456789']
#val_pairs = [pair for pair in all_pairs if pair[3][0] == '9']
#train_pairs = [pair for pair in all_pairs if pair[3][0] in '012345678']

#val_pairs = [pair for pair in all_pairs if pair[4] <= 2000]
#train_pairs = [pair for pair in all_pairs if pair[4] > 2000]

val_pairs = [pair for pair in all_pairs if pair[3] == '11' and pair[4] < 450]
train_pairs = [pair for pair in all_pairs if pair[3] != '11' or pair[4] >= 450]

def vis(pairs):
	for i, pair in enumerate(pairs):
		inputs, mask, valid, targets, _, _ = pair
		skimage.io.imsave('/home/ubuntu/vis/{}_mask.png'.format(i), example[1].astype('uint8')*255)

def count_errors(input_dict, detections1, detections2, outputs, gt):
	def get_array(x):
		m = [(0.0, 0)] * len(detections1)
		for i, sender in enumerate(input_dict['senders']):
			receiver = input_dict['receivers'][i]
			if sender >= len(detections1):
				continue
			output = x[i]
			if output > m[sender][0]:
				m[sender] = (output, receiver)
		return numpy.array([t[1] for t in m], dtype='int32')

	return numpy.count_nonzero(get_array(outputs) - get_array(gt))

print 'initializing model'
m = model.Model([[val_pairs[0][0]], [val_pairs[0][1]]])
session = tf.Session()
session.run(m.init_op)
latest_path = '{}/model_latest/model'.format(PATH)
best_path = '{}/model_best/model'.format(PATH)

print 'begin training'
best_error = None

for epoch in xrange(9999):
	start_time = time.time()
	train_losses = []
	for _ in xrange(1024):
		batch_examples = random.sample(train_pairs, model.BATCH_SIZE)
		d1 = graph_nets.utils_tf.get_feed_dict(m.inputs, graph_nets.utils_np.data_dicts_to_graphs_tuple([example[0] for example in batch_examples]))
		d2 = graph_nets.utils_tf.get_feed_dict(m.targets, graph_nets.utils_np.data_dicts_to_graphs_tuple([example[1] for example in batch_examples]))
		feed_dict = {
			m.input_crops: numpy.concatenate([example[2] for example in batch_examples], axis=0).astype('float32')/255,
			m.is_training: True,
			m.learning_rate: 1e-3,
		}
		feed_dict.update(d1)
		feed_dict.update(d2)
		_, loss = session.run([m.optimizer, m.loss], feed_dict=feed_dict)
		train_losses.append(loss)
	train_loss = numpy.mean(train_losses)
	train_time = time.time()

	val_losses = []
	val_errors = []
	for i in xrange(0, len(val_pairs), model.BATCH_SIZE):
		batch_examples = val_pairs[i:i+model.BATCH_SIZE]
		d1 = graph_nets.utils_tf.get_feed_dict(m.inputs, graph_nets.utils_np.data_dicts_to_graphs_tuple([example[0] for example in batch_examples]))
		d2 = graph_nets.utils_tf.get_feed_dict(m.targets, graph_nets.utils_np.data_dicts_to_graphs_tuple([example[1] for example in batch_examples]))
		feed_dict = {
			m.input_crops: numpy.concatenate([example[2] for example in batch_examples], axis=0).astype('float32')/255,
			m.is_training: False,
		}
		feed_dict.update(d1)
		feed_dict.update(d2)
		loss, outputs = session.run([m.loss, m.outputs], feed_dict=feed_dict)
		val_losses.append(loss)
		val_errors.append(loss)
		#output_counter = 0
		#for j, example in enumerate(batch_examples):
		#	input_dict, target_dict, input_crops, label, frame_idx, detections1, detections2 = example
		#	gt = [t[0] for t in target_dict['edges']]
		#	val_errors.append(count_errors(input_dict, detections1, detections2, outputs[output_counter:output_counter+len(gt), 0], gt))
		#	output_counter += len(gt)

	val_loss = numpy.mean(val_losses)
	val_error = numpy.mean(val_errors)
	val_time = time.time()

	print 'iteration {}: train_time={}, val_time={}, train_loss={}, val_loss={} val_error={}/{}'.format(epoch, int(train_time - start_time), int(val_time - train_time), train_loss, val_loss, val_error, best_error)

	m.saver.save(session, latest_path)
	if best_error is None or val_error < best_error:
		best_error = val_error
		m.saver.save(session, best_path)

def update_image(im, input_dict, detections1, detections2, outputs):
	out = numpy.zeros(im.shape, dtype='uint8')
	out[:, :, :] = im

	m = {}
	for i, sender in enumerate(input_dict['senders']):
		receiver = input_dict['receivers'][i]
		if sender >= len(detections1):
			continue
		output = outputs[i]
		if sender not in m or output > m[sender][0]:
			if receiver == len(detections1):
				m[sender] = (output, None, None)
			else:
				m[sender] = (output, detections1[sender], detections2[receiver - len(detections1) - 1])

	for sender, t in m.items():
		_, p1, p2 = t
		if p1 is None or p2 is None:
			continue
		start = geom.Point(p1[1] * ORIG_HEIGHT / FRAME_SCALE, p1[0] * ORIG_WIDTH / FRAME_SCALE)
		end = geom.Point(p2[1] * ORIG_HEIGHT / FRAME_SCALE, p2[0] * ORIG_WIDTH / FRAME_SCALE)
		for p in geom.draw_line(start, end, geom.Point(im.shape[0], im.shape[1])):
			out[p.x-2:p.x+2, p.y-2:p.y+2, :] = [255, 0, 0]
	return out

def test():
	for i, pair in enumerate(val_pairs[0:64]):
		input_dict, target_dict, input_crops, label, frame_idx, detections1, detections2, num_matches = pair
		gt = [t[0] for t in target_dict['edges']]
		d1 = graph_nets.utils_tf.get_feed_dict(m.inputs, graph_nets.utils_np.data_dicts_to_graphs_tuple([input_dict]))
		feed_dict = {
			m.input_crops: input_crops.astype('float32')/255,
			m.is_training: False,
		}
		feed_dict.update(d1)
		outputs = session.run(m.outputs, feed_dict=feed_dict)[:, 0]
		frame_path = FRAME_PATH.format(label)
		im1 = skimage.io.imread('{}/{}'.format(frame_path, get_frame_fname(frame_idx)))
		im2 = skimage.io.imread('{}/{}'.format(frame_path, get_frame_fname(frame_idx+SKIP[0])))
		skimage.io.imsave('/home/ubuntu/vis/{}_gt.jpg'.format(i), update_image(im1, input_dict, detections1, detections2, gt))
		#skimage.io.imsave('/home/ubuntu/vis/{}_out.jpg'.format(i), update_image(im1, input_dict, detections1, detections2, outputs))
		skimage.io.imsave('/home/ubuntu/vis/{}_b.jpg'.format(i), im2)
