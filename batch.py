import os
for i in range(7,94):
	cmd = 'python3 faceAlign.py rembrandt_lighting_%02d' % i
	os.system(cmd)