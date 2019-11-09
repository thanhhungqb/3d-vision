from pylab import *

import pickle

import numpy
from PIL import Image
from numpy import dot, linalg, array, hstack, cross

import util
from ref import homography, camera, sift

im0_name = 'data/book_frontal.JPG'
im1_name = 'data/book_perspective.JPG'
im0 = array(Image.open(im0_name))
im1 = array(Image.open(im1_name))


def my_calibration(sz):
    row, col = sz
    fx = 2555 * col / 2592
    fy = 2586 * row / 1936
    K = numpy.diag([fx, fy, 1])
    K[0, 2] = 0.5 * col
    K[1, 2] = 0.5 * row
    return K


# compute features
is_process_sift = False
if is_process_sift:
    sift.process_image(im0_name, 'im0.sift')
    sift.process_image(im1_name, 'im1.sift')
l0, d0 = util.read_features_from_file('im0.sift')
l1, d1 = util.read_features_from_file('im1.sift')

# match features and estimate homography
matches = sift.match_twosided(d0, d1)
ndx = matches.nonzero()[0]
fp = homography.make_homog(l0[ndx, :2].T)
ndx2 = [int(matches[i]) for i in ndx]
tp = homography.make_homog(l1[ndx2, :2].T)
model = homography.RansacModel()
H = homography.H_from_ransac(fp, tp, model)[0]  # TODO fix 0

sift.plot_matches(im0, im1, l0, l1, matches)
show()

# ##############################################
print('H', H)
# camera calibration
K = my_calibration((747, 1000))

# 3D points at plane z=0 with sides of length 0.2
box = util.cube_points([0, 0, 0.1], 0.1)

# project bottom square in first image
cam1 = camera.Camera(hstack((K, dot(K, array([[0], [0], [-1]])))))

# first points are the bottom square
box_cam1 = cam1.project(homography.make_homog(box[:, :5]))

# use H to transfer points to the second image
box_trans = homography.normalize(dot(H, box_cam1))

# compute second camera matrix from cam1 and H
cam2 = camera.Camera(dot(H, cam1.P))
A = dot(linalg.inv(K), cam2.P[:, :3])
A = array([A[:, 0], A[:, 1], cross(A[:, 0], A[:, 1])]).T
cam2.P[:, :3] = dot(K, A)

# project with the second camera
box_cam2 = cam2.project(homography.make_homog(box))

# test: projecting point on z=0 should give the same
point = array([1, 1, 0, 1]).T
print homography.normalize(dot(dot(H, cam1.P), point))
print cam2.project(point)

with open('ar_camera.pkl', 'w') as f:
    pickle.dump(K, f)
    pickle.dump(dot(linalg.inv(K), cam2.P), f)

# 2D projection of bottom square
figure()
imshow(im0)
plot(box_cam1[0, :], box_cam1[1, :], linewidth=3)
show()

figure()
imshow(im1)
plot(box_trans[0, :], box_trans[1, :], linewidth=3)
# plot(box_cam2[0, :], box_cam2[1, :], linewidth=3)

show()
