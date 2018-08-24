import os

f = open("keywords.txt", "r")
names = f.read().splitlines()
f.close()

for i in names:
	cmd = 'googleimagesdownload -k "%s" -l 100 -s large' % i
	print(cmd)
	os.system(cmd)