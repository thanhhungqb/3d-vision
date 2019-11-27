import hashlib

from PIL import Image
from matplotlib.pyplot import *
from numpy import *
from numpy.linalg import *

from ref import homography, camera
from ref import sfm
from ref import sift
from util import get_sift_features

if True:
    import os
    import pickle
    from mpl_toolkits.mplot3d import axes3d

    axes3d if False else None

# ----------------------------------------------------
# PREPARING FOR DATA
K = array(
    [
        [2394, 0, 932],
        [0, 2398, 628],
        [0, 0, 1]
    ])

# my K
K = array(
    [
        [2555, 0, 2592//2],
        [0, 2586, 1936//2],
        [0, 0, 1]
    ])

data_dir = 'data'
os.mkdir(data_dir) if not os.path.exists(data_dir) else None
im1_name = '{}/alcatraz1.jpg'.format(data_dir)
im2_name = '{}/alcatraz2.jpg'.format(data_dir)
im1_name = '{}/DSC_2547.jpg'.format(data_dir)
im2_name = '{}/DSC_2548.jpg'.format(data_dir)
# -----------------------------------------------------

# load images and compute features
im1 = array(Image.open(im1_name))
l1, d1 = get_sift_features(im1_name, is_cache=True)

im2 = array(Image.open(im2_name))
l2, d2 = get_sift_features(im2_name, is_cache=True)

# match features
cache_match = "{}/match-{}.data".format(data_dir, hashlib.md5(im1_name + im2_name).hexdigest())
if not os.path.exists(cache_match):
    matches = sift.match_twosided(d1, d2)
    ndx = matches.nonzero()[0]

    with open(cache_match, 'w') as f:
        pickle.dump({'matches': matches, 'ndx': ndx}, f)
else:
    print('use cache match')
    with open(cache_match, 'r') as f:
        c_dic = pickle.load(f)
        matches, ndx = c_dic['matches'], c_dic['ndx']

sift.plot_matches(im1, im2, l1, l2, matches)
show()

# make homogeneous and normalize with inv(K)
x1 = homography.make_homog(l1[ndx, :2].T)
ndx2 = [int(matches[i]) for i in ndx]
x2 = homography.make_homog(l2[ndx2, :2].T)
x1n = dot(inv(K), x1)
x2n = dot(inv(K), x2)

# estimate E with RANSAC, this module take time, so need cache
cache_name = "{}/ransac-{}.data".format(data_dir, hashlib.md5(im1_name + im2_name).hexdigest())
if not os.path.exists(cache_name):
    print('run RanSac, take time')
    model = sfm.RansacModel()
    E, inliers = sfm.F_from_ransac(x1n, x2n, model)

    with open(cache_name, 'w') as f:
        data_dic = {'model': model, 'E': E, 'inliers': inliers}
        pickle.dump(data_dic, f)
else:
    print('use RanSac cache')
    with open(cache_name, 'r') as f:
        data_dic = pickle.load(f)
        model, E, inliers = data_dic['model'], data_dic['E'], data_dic['inliers']

# compute camera matrices (P2 will be list of four solutions)
P1 = array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
P2 = sfm.compute_P_from_essential(E)

# ---------------------------------------------------------
# pick the solution with points in front of cameras
ind = 0
maxres = 0
for i in range(4):
    # triangulate inliers and compute depth for each camera
    X = sfm.triangulate(x1n[:, inliers], x2n[:, inliers], P1, P2[i])
    d1 = dot(P1, X)[2]
    d2 = dot(P2[i], X)[2]
    if sum(d1 > 0) + sum(d2 > 0) > maxres:
        maxres = sum(d1 > 0) + sum(d2 > 0)
        ind = i
        infront = (d1 > 0) & (d2 > 0)

# triangulate inliers and remove points not in front of both cameras
X = sfm.triangulate(x1n[:, inliers], x2n[:, inliers], P1, P2[ind])
X = X[:, infront]

# --------------------------------------------------------------
# 3D plot
fig = figure()
ax = fig.gca(projection='3d')
ax.plot(-X[0], X[1], X[2], 'k.')
axis('off')

# ----------------------------------------------------------------
# plot the projection of X
# project 3D points
cam1 = camera.Camera(P1)
cam2 = camera.Camera(P2[ind])
x1p = cam1.project(X)
x2p = cam2.project(X)
# reverse K normalization
x1p = dot(K, x1p)
x2p = dot(K, x2p)
figure()
imshow(im1)
gray()
# plot(x1p[0], x1p[1], 'o')
# plot(x1[0], x1[1], 'r.')
plot(x1p[1], x1p[0], 'o')
plot(x1[1], x1[0], 'r.')
axis('off')
figure()
imshow(im2)
gray()
# plot(x2p[0], x2p[1], 'o')
# plot(x2[0], x2[1], 'r.')
plot(x2p[1], x2p[0], 'o')
plot(x2[1], x2[0], 'r.')

axis('off')
show()
