from PIL import Image
from pylab import *

import rof
from util import *

im = array(Image.open('empire.jpg').convert('L'))
U50, T = rof.denoise(im, im, tau=0.05)
U150, T = rof.denoise(im, im, tau=0.125)
U250, T = rof.denoise(im, im, tau=0.15)
# U500, T = rof.denoise(im, im, tau=0.5)

gray()
save_3_images('out/tau-125-150-500.png', U50, U150, U250)
