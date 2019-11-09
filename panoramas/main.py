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


N = 5
h = 'lake_500'
featname = ['imgs/{}_{}.sift'.format(h, i) for i in range(N)]
imname = ['imgs/{}_{}.jpg'.format(h, i) for i in range(N)]
l = {}
d = {}
for i in range(5):
    sift.process_image(imname[i], featname[i])
    l[i], d[i] = read_features_from_file(featname[i])

matches = {}
for i in range(4):
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
H_12 = homography.H_from_ransac(fp, tp, model)[0]  # im 1 to 2
fp, tp = convert_points(0)
H_01 = homography.H_from_ransac(fp, tp, model)[0]  # im 0 to 1
tp, fp = convert_points(2)  # NB: reverse order
H_32 = homography.H_from_ransac(fp, tp, model)[0]  # im 3 to 2

tp, fp = convert_points(3)  # NB: reverse order
H_43 = homography.H_from_ransac(fp, tp, model)[0]  # im 4 to 3

# warp the images

delta = 2000  # for padding and translation
im1 = array(Image.open(imname[1]))
im2 = array(Image.open(imname[2]))
im_12 = warp.panorama(H_12, im1, im2, delta, delta)
imsave('imgs/{}-out-12.png'.format(h), im_12)

im1 = array(Image.open(imname[0]))
im_02 = warp.panorama(dot(H_12, H_01), im1, im_12, delta + delta)
# im_02 = warp.panorama(dot(H_01, H_12), im1, im_12, delta, delta)
imsave('imgs/{}-out-012.png'.format(h), im_02)

im1 = array(Image.open(imname[3]))
im_32 = warp.panorama(H_32, im1, im_02, delta, delta)
imsave('imgs/{}-out-0123.png'.format(h), im_32)

im1 = array(Image.open(imname[3 + 1]))
im_42 = warp.panorama(dot(H_32, H_43), im1, im_32, delta, 2 * delta)

imsave('imgs/{}-out-42.png'.format(h), im_42)
