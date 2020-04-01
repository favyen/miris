import model

import json
import random
import sys
import tensorflow as tf

def eprint(s):
	sys.stderr.write(s + '\n')
	sys.stderr.flush()

def oprint(s):
	sys.stdout.write(s + '\n')
	sys.stdout.flush()

mode = sys.argv[1]
model_path = sys.argv[2]
NORM_SIZE = 1000.0

m = model.Model()
session = tf.Session()
m.saver.restore(session, model_path)

def get_data(detection):
	cx = (detection['left']+detection['right'])/2/NORM_SIZE
	cy = (detection['top']+detection['bottom'])/2/NORM_SIZE
	width = (detection['right'] - detection['left'])/NORM_SIZE
	height = (detection['bottom']-detection['top'])/NORM_SIZE
	return [cx, cy, width, height]

def pad_track(track):
	if track is None:
		track = []
	if len(track) > model.MAX_LENGTH:
		track = random.sample(track, model.MAX_LENGTH)
		track.sort(key=lambda det: det['frame_idx'])
	data = [get_data(det) for det in track]
	l = len(data)
	while len(data) < model.MAX_LENGTH:
		data.append([0, 0, 0, 0])
	return data, l

while True:
	line = sys.stdin.readline()
	if not line:
		break
	tracks = json.loads(line.strip())
	outputs = []
	for i in xrange(0, len(tracks), model.BATCH_SIZE):
		eprint('... {}/{}'.format(i, len(tracks)))
		batch = tracks[i:i+model.BATCH_SIZE]
		batch = [pad_track(track) for track in batch]
		batch_outputs = session.run(m.outputs, feed_dict={
			m.is_training: False,
			m.inputs: [t[0] for t in batch],
			m.lengths: [t[1] for t in batch],
		})
		outputs.extend(batch_outputs.tolist())
	s = json.dumps(outputs)
	oprint(s)
