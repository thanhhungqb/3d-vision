from PIL import Image
from pylab import *
from scipy.ndimage import filters

import rof
from util import *

im = array(Image.open('empire.jpg').convert('L'))
U, T = rof.denoise(im, im)
# figure()
gray()
# # imshow(U)
# axis('equal')
# axis('off')
# show()

G = filters.gaussian_filter(im, 10)

save_3_images('out/empire.png', im, G, U)
