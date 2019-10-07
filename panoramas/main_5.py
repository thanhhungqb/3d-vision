# function to convert the matches to hom. points
from PIL import Image
from numpy import *
from scipy.misc import imsave

import util
from ref import sift, homography, warp


def homo2(im1, im2, delta=2000):
    """
    homo 2 images given path, im1 in left of im2
    :param im1:
    :param im2:
    :return:
    """
    print('homo for', im1, im2)
    imname = [im1, im2]
    featname = ["{}.sift".format(o) for o in imname]
    l, d = {}, {}
    for i in range(2):
        sift.process_image(imname[i], featname[i])
        l[i], d[i] = util.read_features_from_file(featname[i])

    matches = {}
    matches[0] = sift.match(d[1], d[0])

    model = homography.RansacModel()

    fp, tp = util.convert_points(matches, l, 0)
    H_01 = homography.H_from_ransac(fp, tp, model)[0]  # im 0 to 1

    # homo
    im1 = array(Image.open(imname[0]))
    im2 = array(Image.open(imname[1]))

    out = warp.panorama(H_01, im1, im2, delta, delta)
    out_name = '{}-out.png'.format(imname[0])
    imsave(out_name, out)
    return out_name


# N = 3
h = 'lake_500'
# ((((0, 1), 2), 3), 4) order
# o01 = homo2('imgs/{}_{}.jpg'.format(h, 0), 'imgs/{}_{}.jpg'.format(h, 1))
# o012 = homo2(o01, 'imgs/{}_{}.jpg'.format(h, 2))
# o0123 = homo2(o012, 'imgs/{}_{}.jpg'.format(h, 3))
# o01234 = homo2(o0123, 'imgs/{}_{}.jpg'.format(h, 4))

# (((0, (1, 2)), 3), 4) order FAIL
# o12 = homo2('imgs/{}_{}.jpg'.format(h, 1), 'imgs/{}_{}.jpg'.format(h, 2))
# o012 = homo2('imgs/{}_{}.jpg'.format(h, 0), o12)
# o34 = homo2('imgs/{}_{}.jpg'.format(h, 3), 'imgs/{}_{}.jpg'.format(h, 4))
# o01234 = homo2(o012, o34)

# ((0, (1, 2)), (3, 4)) order
o12 = homo2('imgs/{}_{}.jpg'.format(h, 1), 'imgs/{}_{}.jpg'.format(h, 2), delta=500)
o012 = homo2('imgs/{}_{}.jpg'.format(h, 0), o12, delta=1000)
o0123 = homo2(o012, 'imgs/{}_{}.jpg'.format(h, 3), delta=2000)
o01234 = homo2(o0123, 'imgs/{}_{}.jpg'.format(h, 4), delta=1000)

# featname = ['imgs/{}_{}.sift'.format(h, i + 1) for i in range(N)]
# imname = ['imgs/{}_{}.jpg'.format(h, i + 1) for i in range(N)]
# l = {}
# d = {}
# for i in range(N):
#     sift.process_image(imname[i], featname[i])
#     l[i], d[i] = util.read_features_from_file(featname[i])
#
# matches = {}
# for i in range(N - 1):
#     matches[i] = sift.match(d[i + 1], d[i])
#
# # estimate the homographies
# model = homography.RansacModel()
# fp, tp = util.convert_points(matches, l, 1)
# # H_12 = homography.H_from_ransac(fp, tp, model)[0]  # im 1 to 2
# H_12 = homography.H_from_ransac(tp, fp, model)[0]  # im 2 to 1 (Reverse)
#
# fp, tp = util.convert_points(matches, l, 0)
# H_01 = homography.H_from_ransac(fp, tp, model)[0]  # im 0 to 1
#
# # warp the images
#
# delta = 2000  # for padding and translation
# im1 = array(Image.open(imname[1]))
# im2 = array(Image.open(imname[2]))
# # im_12 = warp.panorama(H_12, im1, im2, delta, delta)
# im_12 = warp.panorama(H_12, im2, im1, delta, delta)
# imsave('imgs/{}-out-3-12.png'.format(h), im_12)
#
# im1 = array(Image.open(imname[0]))
# # im_02 = warp.panorama(dot(H_12, H_01), im1, im_12, delta, delta)
# im_02 = warp.panorama(H_01, im1, im_12, delta, delta)
# imsave('imgs/{}-out-3-012.png'.format(h), im_02)
