import pickle

import pygame
import pygame.image
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from numpy import linalg, dot, eye, array
from numpy.ma import arctan
from pygame import *
from scipy.constants import pi


def set_projection_from_camera(K):
    """ Set view from a camera calibration matrix. """
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    fx = K[0, 0]
    fy = K[1, 1]
    fovy = 2 * arctan(0.5 * height / fy) * 180 / pi
    aspect = (width * fy) / (height * fx)
    # define the near and far clipping planes
    near = 0.1
    far = 100.0
    # set perspective
    gluPerspective(fovy, aspect, near, far)
    glViewport(0, 0, width, height)


def set_modelview_from_camera(Rt):
    """ Set the model view matrix from camera pose. """
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    # rotate teapot 90 deg around x-axis so that z-axis is up
    Rx = array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    # set rotation to best approximation
    R = Rt[:, :3]
    U, S, V = linalg.svd(R)
    R = dot(U, V)
    R[0, :] = -R[0, :]  # change sign of x-axis
    # set translation

    t = Rt[:, 3]
    # setup 4*4 model view matrix
    M = eye(4)
    M[:3, :3] = dot(R, Rx)
    M[:3, 3] = t
    # transpose and flatten to get column order
    M = M.T
    m = M.flatten()
    # replace model view with the new matrix
    glLoadMatrixf(m)


def draw_background(imname):
    """ Draw background image using a quad. """
    # load background image (should be .bmp) to OpenGL texture
    bg_image = pygame.image.load(imname).convert()
    bg_data = pygame.image.tostring(bg_image, "RGBX", 1)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    # bind the texture
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, glGenTextures(1))
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, bg_data)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    # create quad to fill the whole window
    glBegin(GL_QUADS)
    glTexCoord2f(0.0, 0.0);
    glVertex3f(-1.0, -1.0, -1.0)
    glTexCoord2f(1.0, 0.0);
    glVertex3f(1.0, -1.0, -1.0)
    glTexCoord2f(1.0, 1.0);
    glVertex3f(1.0, 1.0, -1.0)
    glTexCoord2f(0.0, 1.0);
    glVertex3f(-1.0, 1.0, -1.0)
    glEnd()
    # clear the texture
    glDeleteTextures(1)


def draw_teapot(size):
    """ Draw a red teapot at the origin. """
    glutInit()
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_DEPTH_TEST)
    glClear(GL_DEPTH_BUFFER_BIT)
    # draw red teapot
    glMaterialfv(GL_FRONT, GL_AMBIENT, [0, 0, 0, 0])
    glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.5, 0.0, 0.0, 0.0])
    glMaterialfv(GL_FRONT, GL_SPECULAR, [0.7, 0.6, 0.6, 0.0])
    glMaterialf(GL_FRONT, GL_SHININESS, 0.25 * 128.0)
    glutSolidTeapot(size)


# glutSolidTeapot(45)

def draw_im(path, K, Rt, tea_size=0.04):
    draw_background(path)
    set_projection_from_camera(K)
    set_modelview_from_camera(Rt)
    draw_teapot(tea_size)


def setup():
    """ Setup window and pygame environment. """
    pygame.init()
    win = pygame.display.set_mode((width, height), OPENGL | DOUBLEBUF)
    pygame.display.set_caption('OpenGL AR demo')
    return win


# load camera data
with open('ar_camera.pkl', 'r') as f:
    K = pickle.load(f)
    Rt1 = pickle.load(f)
    Rt0 = pickle.load(f)

# draw_background('data/book_frontal.JPG')
# draw_background('data/book_perspective.bmp')
# set_projection_from_camera(K)
# set_modelview_from_camera(Rt)
# set_modelview_from_camera(Rt1)
# draw_teapot(0.02)

im0_name = 'data/book_frontal.JPG'
im1_name = 'data/book_perspective.JPG'
im0_name = 'data/d1-2.JPG'
im1_name = 'data/d2-2.JPG'
im1_name = 'data/d3-2.JPG'
width, height = 1000, 747

wh = 1215, 1634
width, height = wh
# width, height = 747, 100


print(K, Rt0, Rt1)

win = setup()

# draw_im(im0_name, K, Rt0)
draw_im(im1_name, K, Rt1)

# while True:
#     # pass
#     event = pygame.event.poll()
#     if event.type in (QUIT, KEYDOWN):
#         break
#     pygame.display.flip()

pygame.image.save(win, "screenshot.jpeg")
