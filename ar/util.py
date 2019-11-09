from PIL import Image
from pylab import *

from ref import homography


def sift_show(imagename, resultname):
    """ Process an image and save the results in a file. """

    if imagename[-3:] != 'pgm':
        # create a pgm file
        im = Image.open(imagename).convert('L')
        im.save('tmp.pgm')
        imagename = 'tmp.pgm'

    cmmd = str("./sift -display <" + imagename + " >" + resultname)

    os.system(cmmd)
    print 'processed', imagename, 'to', resultname


def read_features_from_file(filename):
    """ Read feature properties and return in matrix form. """

    with open(filename, "r") as f:
        contents = f.read()
    a = contents.split()
    a = [float(o) for o in a]
    a = array(a[2:])  # remove 2 two index
    a = a.reshape((-1, 4 + 128))
    return a[:, :4], a[:, 4:]  # feature locations, descriptors


def convert_points(matches, l, j):
    ndx = matches[j].nonzero()[0]

    fp = homography.make_homog(l[j + 1][ndx, :2].T)
    ndx2 = [int(matches[j][i]) for i in ndx]
    tp = homography.make_homog(l[j][ndx2, :2].T)
    return fp, tp


def cube_points(c, wid):
    """ Creates a list of points for plotting
    a cube with plot. (the first 5 points are
    the bottom square, some sides repeated). """
    p = []
    # bottom
    p.append([c[0] - wid, c[1] - wid, c[2] - wid])
    p.append([c[0] - wid, c[1] + wid, c[2] - wid])
    p.append([c[0] + wid, c[1] + wid, c[2] - wid])
    p.append([c[0] + wid, c[1] - wid, c[2] - wid])
    p.append([c[0] - wid, c[1] - wid, c[2] - wid])  # same as first to close plot
    # top
    p.append([c[0] - wid, c[1] - wid, c[2] + wid])
    p.append([c[0] - wid, c[1] + wid, c[2] + wid])
    p.append([c[0] + wid, c[1] + wid, c[2] + wid])
    p.append([c[0] + wid, c[1] - wid, c[2] + wid])
    p.append([c[0] - wid, c[1] - wid, c[2] + wid])  # same as first to close plot
    # vertical sides
    p.append([c[0] - wid, c[1] - wid, c[2] + wid])
    p.append([c[0] - wid, c[1] + wid, c[2] + wid])
    p.append([c[0] - wid, c[1] + wid, c[2] - wid])
    p.append([c[0] + wid, c[1] + wid, c[2] - wid])
    p.append([c[0] + wid, c[1] + wid, c[2] + wid])
    p.append([c[0] + wid, c[1] - wid, c[2] + wid])
    p.append([c[0] + wid, c[1] - wid, c[2] - wid])
    return array(p).T


if __name__ == '__main__':
    sift_show('imgs/lake_500_1.jpg', 'imgs/lake_500_1.sift.pgm')
    sift_show('imgs/lake_500_2.jpg', 'imgs/lake_500_2.sift.pgm')
    sift_show('imgs/lake_500_3.jpg', 'imgs/lake_500_3.sift.pgm')
