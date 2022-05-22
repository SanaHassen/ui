from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
from math import sqrt
parser = argparse.ArgumentParser(description='Code for AKAZE local features matching tutorial.')
parser.add_argument('--input1', help='Path to input image 1.', default='template.png')
parser.add_argument('--input2', help='Path to input image 2.', default='original.png')
parser.add_argument('--homography', help='Path to the homography matrix.', default='H1to3p.xml')
args = parser.parse_args()
img1 = cv.imread(cv.samples.findFile(args.input1), cv.IMREAD_GRAYSCALE)
img2 = cv.imread(cv.samples.findFile(args.input2), cv.IMREAD_GRAYSCALE)
if img1 is None or img2 is None:
    print('Could not open or find the images!')
    exit(0)
fs = cv.FileStorage(cv.samples.findFile(args.homography), cv.FILE_STORAGE_READ)
homography = fs.getFirstTopLevelNode().mat()