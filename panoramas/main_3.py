# function to convert the matches to hom. points
from PIL import Image
from numpy import *
from scipy.misc import imsave

import util
from ref import sift, homography, warp

N = 3
h = 'lake_500'
featname = ['imgs/{}_{}.sift'.format(h, i + 1) for i in range(N)]
imname = ['imgs/{}_{}.jpg'.format(h, i + 1) for i in range(N)]
l = {}
d = {}
for i in range(N):
    sift.process_image(imname[i], featname[i])
    l[i], d[i] = util.read_features_from_file(featname[i])

matches = {}
for i in range(N - 1):
    matches[i] = sift.match(d[i + 1], d[i])

# estimate the homographies
model = homography.RansacModel()
fp, tp = util.convert_points(matches, l, 1)
# H_12 = homography.H_from_ransac(fp, tp, model)[0]  # im 1 to 2
H_12 = homography.H_from_ransac(tp, fp, model)[0]  # im 2 to 1 (Reverse)

fp, tp = util.convert_points(matches, l, 0)
H_01 = homography.H_from_ransac(fp, tp, model)[0]  # im 0 to 1

# warp the images

delta = 2000  # for padding and translation
im1 = array(Image.open(imname[1]))
im2 = array(Image.open(imname[2]))
# im_12 = warp.panorama(H_12, im1, im2, delta, delta)
im_12 = warp.panorama(H_12, im2, im1, delta, delta)
imsave('imgs/{}-out-3-12.png'.format(h), im_12)

im1 = array(Image.open(imname[0]))
# im_02 = warp.panorama(dot(H_12, H_01), im1, im_12, delta, delta)
im_02 = warp.panorama(H_01, im1, im_12, delta, delta)
imsave('imgs/{}-out-3-012.png'.format(h), im_02)
