import hashlib

from PIL import Image
from matplotlib.pyplot import *
from numpy import *

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
# show()

# make homogeneous and normalize with inv(K)
x1 = homography.make_homog(l1[ndx, :2].T)
ndx2 = [int(matches[i]) for i in ndx]
x2 = homography.make_homog(l2[ndx2, :2].T)

# estimate F
F = sfm.compute_fundamental(x1, x2)
P1 = array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
P2 = sfm.compute_P_from_fundamental(F)


# ---------------------------------------------------------
# pick the solution with points in front of cameras
def refineX(X):
    x1 = X[1]
    mean = x1.mean()
    std = x1.std()
    idx = [i for i in range(len(x1)) if mean - 2 * std < x1[i] < mean + 2 * std]

    x2 = X[2]
    mean, std = x2.mean(), x2.std()
    idx = [i for i in idx if mean - 2 * std < x2[i] < mean + 2 * std]

    x0 = X[0]
    mean, std = x0.mean(), x0.std()
    idx = [i for i in idx if mean - 2 * std < x0[i] < mean + 2 * std]

    x1 = X[1][idx]
    x2 = X[2][idx]
    x3 = X[3][idx]
    x0 = X[0][idx]
    nX = np.array([x0, x1, x2, x3])
    return nX


X = sfm.triangulate(x1, x2, P1, P2)
lX = len(X[0])
X = refineX(X)
print('test', lX, len(X[0]))
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
cam2 = camera.Camera(P2)
x1p = cam1.project(X)
x2p = cam2.project(X)

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
