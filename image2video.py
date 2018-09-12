#!/usr/local/bin/python3

import cv2
import argparse
import os

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-fn", "--foldername", required=True, help="folder name needed.")
ap.add_argument("-o", "--output", required=False, default='output.mp4', help="output video file")
args = vars(ap.parse_args())

# Arguments
dir_path = args['foldername']
output = args['output']

images = []
for f in os.listdir(dir_path):
	images.append(f)

# Determine the width and height from the first image
image_path = os.path.join(dir_path, images[0])
frame = cv2.imread(image_path)
height, width, channels = frame.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
out = cv2.VideoWriter(output, fourcc, 60.0, (width, height))

numOfImage = len(images)
for idx, image in enumerate(images):
	if idx % 100 == 0:
		print("Processing %d of %d ..." % (idx, numOfImage)) 

	image_path = os.path.join(dir_path, image)
	frame = cv2.imread(image_path)

	out.write(frame) # Write out frame to video

# Release everything if job is finished
out.release()

print("The output video is {}".format(output))