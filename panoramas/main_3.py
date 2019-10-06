# function to convert the matches to hom. points
from PIL import Image
from numpy import *
from scipy.misc import imsave

from ref import sift, homography, warp


def read_features_from_file(filename):
    """ Read feature properties and return in matrix form. """

    with open(filename, "r") as f:
        contents = f.read()
    a = contents.split()
    a = [float(o) for o in a]
    a = array(a[2:])  # remove 2 two index
    a = a.reshape((-1, 4 + 128))
    return a[:, :4], a[:, 4:]  # feature locations, descriptors


N = 3
h = 'lake_500'
featname = ['imgs/{}_{}.sift'.format(h, i + 1) for i in range(N)]
imname = ['imgs/{}_{}.jpg'.format(h, i + 1) for i in range(N)]
l = {}
d = {}
for i in range(N):
    sift.process_image(imname[i], featname[i])
    l[i], d[i] = read_features_from_file(featname[i])

matches = {}
for i in range(N - 1):
    matches[i] = sift.match(d[i + 1], d[i])


def convert_points(j):
    ndx = matches[j].nonzero()[0]

    fp = homography.make_homog(l[j + 1][ndx, :2].T)
    ndx2 = [int(matches[j][i]) for i in ndx]
    tp = homography.make_homog(l[j][ndx2, :2].T)
    return fp, tp


# estimate the homographies
model = homography.RansacModel()
fp, tp = convert_points(1)
# H_12 = homography.H_from_ransac(fp, tp, model)[0]  # im 1 to 2
H_12 = homography.H_from_ransac(tp, fp, model)[0]  # im 2 to 1 (Reverse)

fp, tp = convert_points(0)
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
