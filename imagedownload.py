import os

f = open("2017beauty.txt", "r")
names = f.read().splitlines()
f.close()

for i in names:
	cmd = 'googleimagesdownload -k "%s" -l 100 -s large' % i
	print(cmd)
	os.system(cmd)