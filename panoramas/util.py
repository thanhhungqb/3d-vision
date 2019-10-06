from PIL import Image
from pylab import *


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


if __name__ == '__main__':
    sift_show('imgs/lake_500_1.jpg', 'imgs/lake_500_1.sift.pgm')
    sift_show('imgs/lake_500_2.jpg', 'imgs/lake_500_2.sift.pgm')
    sift_show('imgs/lake_500_3.jpg', 'imgs/lake_500_3.sift.pgm')
