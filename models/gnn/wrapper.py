import geom
import model

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import graph_nets
import json
import numpy
import math
import os.path
import skimage.io
import sys
import tensorflow as tf
import time

def eprint(s):
	sys.stderr.write(s + '\n')
	sys.stderr.flush()

def oprint(s):
	sys.stdout.write(s + '\n')
	sys.stdout.flush()

def pad6(x):
	s = str(x)
	while len(s) < 6:
		s = '0' + s
	return s

model_path = sys.argv[1]
detection_path = sys.argv[2]
frame_path = sys.argv[3]
frame_scale = int(sys.argv[4])

ORIG_WIDTH = 1920
ORIG_HEIGHT = 1080
CROP_SIZE = 64

eprint('initializing model')
input_example = {
	'globals': [],
	'nodes': [[1.0] * (8+64), [1.0] * (8+64)],
	'edges': [[1.0] * 6, [1.0] * 6],
	'senders': [0, 1],
	'receivers': [1, 0],
}
target_example = {
	'globals': [],
	'nodes': [[], []],
	'edges': [[0.0], [1.0]],
	'senders': [0, 1],
	'receivers': [1, 0],
}
m = model.Model([[input_example], [target_example]])
config = tf.ConfigProto(
	device_count={'GPU': 0}
)
session = tf.Session(config=config)
m.saver.restore(session, model_path)

eprint('loading detections from {}'.format(detection_path))
with open(detection_path, 'r') as f:
	detections = json.load(f)

def zip_frame_info(detections, frame_idx):
	im = skimage.io.imread('{}/{}.jpg'.format(frame_path, pad6(frame_idx)))
	im_bounds = geom.Rectangle(
		geom.Point(0, 0),
		geom.Point(im.shape[0], im.shape[1])
	)
	info = []
	for idx, detection in enumerate(detections):
		rect = geom.Rectangle(
			geom.Point(detection['top']/frame_scale, detection['left']/frame_scale),
			geom.Point(detection['bottom']/frame_scale, detection['right']/frame_scale)
		)
		rect = im_bounds.clip_rect(rect)
		if rect.lengths().x < 4 or rect.lengths().y < 4:
			continue
		crop = im[rect.start.x:rect.end.x, rect.start.y:rect.end.y, :]
		resize_factor = min([float(CROP_SIZE) / crop.shape[0], float(CROP_SIZE) / crop.shape[1]])
		crop = (skimage.transform.resize(crop, [int(crop.shape[0] * resize_factor), int(crop.shape[1] * resize_factor)])*255).astype('uint8')
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
	senders = []
	receivers = []
	input_nodes = []
	input_edges = []
	input_crops = numpy.zeros((len(info1)+1+len(info2), CROP_SIZE, CROP_SIZE, 3), dtype='uint8')

	for i, t in enumerate(info1):
		detection, crop, _ = t
		cx, cy = get_loc(detection)
		input_nodes.append([cx, cy, detection['width'], detection['height'], 1, 0, 0, skip/50.0] + [0.0]*64)
		input_crops[i, :, :, :] = crop
	input_nodes.append([0.5, 0.5, 0, 0, 0, 1, 0] + [0.0]*64)
	for i, t in enumerate(info2):
		detection, crop, _ = t
		cx, cy = get_loc(detection)
		input_nodes.append([cx, cy, detection['width'], detection['height'], 0, 0, 1, skip/50.0] + [0.0]*64)
		input_crops[len(info1)+1+i, :, :, :] = crop

	for i, t1 in enumerate(info1):
		detection1, _, _ = t1
		x1, y1 = get_loc(detection1)

		for j, t2 in enumerate(info2):
			detection2, _, _ = t2
			x2, y2 = get_loc(detection2)

			senders.extend([i, len(info1) + 1 + j])
			receivers.extend([len(info1) + 1 + j, i])
			edge_shared = [x2 - x1, y2 - y1, math.sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1))]
			input_edges.append(edge_shared + [1, 0, 0])
			input_edges.append(edge_shared + [0, 1, 0])

		senders.extend([i, len(info1)])
		receivers.extend([len(info1), i])
		edge_shared = [0.0, 0.0, 0.0]
		input_edges.append(edge_shared + [0, 0, 0])
		input_edges.append(edge_shared + [1, 0, 0])

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
	add_internal_edges(info1, 0)
	add_internal_edges(info2, len(info1) + 1)

	input_dict = {
		"globals": [],
		"nodes": input_nodes,
		"edges": input_edges,
		"senders": senders,
		"receivers": receivers,
	}
	return input_dict, input_crops

eprint('finished loading')
times = {'read': 0, 'run': 0, 'count': 0}
while True:
	line = sys.stdin.readline()
	if not line:
		break
	# list of (idx1, idx2)
	indices = json.loads(line.strip())

	t0 = time.time()
	frame_infos = {}
	for frame_t in indices:
		for frame_idx in frame_t:
			if frame_idx in frame_infos:
				continue
			elif not detections[frame_idx]:
				frame_infos[frame_idx] = []
				continue
			frame_infos[frame_idx] = zip_frame_info(detections[frame_idx], frame_idx)

	t1 = time.time()
	mats = []
	for idx1, idx2 in indices:
		info1 = frame_infos[idx1]
		info2 = frame_infos[idx2]

		if len(info1) == 0 or len(info2) == 0:
			mats.append(numpy.zeros((len(info1), len(info2)+1), dtype='float32').tolist())
			continue

		input_dict, input_crops = get_frame_pair(info1, info2, idx2-idx1)
		d1 = graph_nets.utils_tf.get_feed_dict(m.inputs, graph_nets.utils_np.data_dicts_to_graphs_tuple([input_dict]))
		feed_dict = {
			m.input_crops: input_crops.astype('float32')/255,
			m.is_training: False,
		}
		feed_dict.update(d1)
		outputs = session.run(m.outputs, feed_dict=feed_dict)[:, 0]
		mat = numpy.zeros((len(detections[idx1]), len(detections[idx2])+1), dtype='float32')
		for i, sender in enumerate(input_dict['senders']):
			receiver = input_dict['receivers'][i]
			if sender >= len(info1) or receiver < len(info1):
				continue
			_, _, s_idx = info1[sender]
			if receiver == len(info1):
				r_idx = len(detections[idx2])
			else:
				_, _, r_idx = info2[receiver - len(info1) - 1]
			mat[s_idx, r_idx] = outputs[i]

		mats.append(mat.tolist())

	t2 = time.time()
	times['read'] += t1-t0
	times['run'] += t2-t1
	times['count'] += 1
	if times['count'] % 128 == 0:
		eprint(str(times))

	s = json.dumps(mats)
	oprint(s)
