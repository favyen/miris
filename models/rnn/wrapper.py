import model

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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

num_outputs = int(sys.argv[1])
model_path = sys.argv[2]

m = model.Model(num_outputs=num_outputs)
config = tf.ConfigProto(
	device_count={'GPU': 0}
)
session = tf.Session(config=config)
m.saver.restore(session, model_path)

while True:
	line = sys.stdin.readline()
	if not line:
		break
	tracks = json.loads(line.strip())
	outputs = []
	for i in range(0, len(tracks), model.BATCH_SIZE):
		if i % 1024 == 0:
			eprint('... {}/{}'.format(i, len(tracks)))
		batch = tracks[i:i+model.BATCH_SIZE]
		batch = [model.pad_track(track) for track in batch]
		batch_outputs = session.run(m.outputs, feed_dict={
			m.is_training: False,
			m.inputs: [t[0] for t in batch],
			m.lengths: [t[1] for t in batch],
		})
		outputs.extend(batch_outputs.tolist())
	s = json.dumps(outputs)
	oprint(s)
