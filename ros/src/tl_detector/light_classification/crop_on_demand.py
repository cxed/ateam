#!/usr/bin/env python
import cv2
import argparse
import random

# Reads a full camera image. Writes to the same directory 85 cropped images taken from various
# locations in the full image.

# Basically, get the full image tar file.
#     $ tar -xvzf tl_imgs.tgz
#     $ cd tl_imgs/Y
#     $ ../../crop_on_demand.py *png
#     $ rm B?????.png  # Get rid of originals and leave only the cropped ones.
#     $ cd ../G
#     $ ../../crop_on_demand.py *png
#     $ rm B?????.png
#     $ cd ../R
#     $ ../../crop_on_demand.py *png
#     $ rm B?????.png
# Then you're ready to use a zillion high quality images for training.
# You can also modify the script to crop out other regions you might want to explore.

ap= argparse.ArgumentParser()
#ap.add_argument('-p','--prefix',required=False,help='Output filename prefix',default="crop_")
ap.add_argument('images', metavar='file', type=str, nargs='+', help='One or more filenames.')
A= ap.parse_args()

def write_crop(ofn,i,x,y):
    '''Writes to filename (n) a 200x300 image (i) centered at the supplied coordinates (x,y).'''
    cropi= i.copy()
    cropi= cropi[(y-225):(y+225),(x-150):(x+150)]
    cropi= cv2.resize(cropi,(200,300),interpolation = cv2.INTER_CUBIC)
    cv2.imwrite(ofn,cropi)

for ifn in A.images:
    c=0
    print("Processing: %s"%ifn)
    i= cv2.imread(ifn)
    x,y= 400,300 
    write_crop(ifn[:-4]+('_crop%02d.png'%c),i,x,y)
    for yoffset in range(-40,60,20):
        for xoffset in range(-200,225,25):
            r= random.randint(-20,20)
            write_crop(ifn[:-4]+('_crop%02d.png'%c),i,x+xoffset+r,y+yoffset+r)
            c+=1
