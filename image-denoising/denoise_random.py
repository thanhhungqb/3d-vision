from matplotlib.pyplot import gray
from numpy import *
from numpy import random
from scipy.ndimage import filters

import rof
from util import *

# create synthetic image with noise
im = zeros((500, 500))
im[100:400, 100:400] = 128
im[200:300, 200:300] = 255
im = im + 30 * random.standard_normal((500, 500))
U, T = rof.denoise(im, im)
G = filters.gaussian_filter(im, 10)

# save the result
gray()
save_3_images('out/random.png', im, G, U)
