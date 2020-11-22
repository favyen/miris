import json
import os
import subprocess

def pad6(x):
	s = str(x)
	while len(s) < 6:
		s = '0'+s
	return s

def f(ds_list, x_list):
	for ds in ds_list:
		for x in x_list:
			frame_path = 'data/{}/frames/{}/'.format(ds, x)
			subprocess.call(['mkdir', '-p', frame_path])
			subprocess.call(['ffmpeg', '-i', 'data/{}/videos/{}.mp4'.format(ds, x), '-q:v', '1', frame_path+'%06d.jpg'])

			# if there are detections on frame 0, it means we need to re-number the frames from 000001 -> 000000
			with open('data/{}/json/{}-baseline.json'.format(ds, x), 'r') as f:
				detections = json.load(f)
			if not detections[0]:
				continue

			for frame_idx in range(len(detections)):
				src_fname = frame_path+pad6(frame_idx+1)+'.jpg'
				dst_fname = frame_path+pad6(frame_idx)+'.jpg'
				os.rename(src_fname, dst_fname)

f(['shibuya', 'beach', 'warsaw'], range(6))
f(['uav'], ['0006', '0007', '0008', '0009', '0011'])
