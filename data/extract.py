import subprocess

for ds in ['shibuya', 'beach', 'warsaw']:
	for x in range(6):
		subprocess.call(['mkdir', '-p', 'data/{}/frames/{}/'.format(ds, x)])
		subprocess.call(['ffmpeg', '-i', 'data/{}/videos/{}.mp4'.format(ds, x), '-q:v', '1', 'data/{}/frames/{}/%06d.jpg'.format(ds, x)])
