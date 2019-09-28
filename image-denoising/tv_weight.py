from PIL import Image
from pylab import *

import rof
from util import *

im = array(Image.open('empire.jpg').convert('L'))
U1, T = rof.denoise(im, im, tv_weight=1)
U10, T = rof.denoise(im, im, tv_weight=10)
U100, T = rof.denoise(im, im, tv_weight=100)
gray()
# G = filters.gaussian_filter(im, 5)

save_3_images('out/tv-weight-1-10-100.png', U1, U10, U100)
