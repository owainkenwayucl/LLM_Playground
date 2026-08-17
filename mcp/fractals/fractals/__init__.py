'''
   Tools for generating fractals.
   Owain Kenway, 2023, 2026
'''

import numpy
import os
import numba
import io
import base64
from PIL import Image

MAX_ITERATIONS=1000
NEXT_PLOT_NUM=0

# Wether or not to output information to the terminal when running.
PRINT_MESSAGES=True

# Have constantly updating filename
def NEXT_PLOT(suffix=''):
    global NEXT_PLOT_NUM
    NEXT_PLOT_NUM += 1
    dot_suffix = ''
    if not suffix == '':
        dot_suffix = '.' + suffix
    ret_val = 'output' + os.sep + 'output_' + str(NEXT_PLOT_NUM) + dot_suffix
    if os.path.isfile(ret_val):
        return NEXT_PLOT(suffix)
    return ret_val

# Function for Mandelbrot sets.
@numba.jit(nopython=True)
def mandel(c, max_iter=MAX_ITERATIONS):
    iterations = 0
    z = 0 + 0j
    while (((numpy.absolute(z)) < 2) and (iterations < max_iter)):
        z = (z**2) + c
        iterations = iterations + 1
    return iterations

# Generator function for Julia set functions from given c, n.
def generate_julia(c, n):
    @numba.jit(nopython=True)
    def julia(z, max_iter=MAX_ITERATIONS):
        iterations = 0
        while (((numpy.absolute(z)) < 2) and (iterations < max_iter)):
            z = (z**n) + c
            iterations = iterations + 1
        return iterations
    return julia

# Generate an image (numpy array) of iterations for a given size, function, range, and maximum iterations.
@numba.jit(nopython=True, parallel=True, nogil=True)
def generate_fractal(width, height, func, xmin=-2, xmax=1, ymin=-1, ymax=1, max_iter=MAX_ITERATIONS):
    image = numpy.zeros((width, height), dtype=numpy.int64)
    xvals = numpy.linspace(xmin, xmax, width)
    yvals = numpy.linspace(ymin, ymax, height)
    for py in numba.prange(height):
        for px in range(width):
            c = xvals[px] + (1j*yvals[py])
            image[px,height - py - 1] = func(c,max_iter)
    return (image, max_iter + 1)

# Convert image data into 255 levels.
def normalise_greyscale(image_data):
    imax = image_data[0].max()
    imin = image_data[0].min()

    if imax == imin:
        return numpy.zeros(image_data[0].shape, dtype=numpy.uint8)

    if imax < image_data[1]:
        imax = image_data[1]

    if imin > 0:
        imin = 0

    nimg = (image_data[0].astype(float) - imin) / (imax - imin) * 255
    return nimg.astype(numpy.uint8)

# Write image data as a file
def write_image_pillow(image_data, filename=None):
    image = Image.fromarray(numpy.flipud(numpy.rot90(normalise_greyscale(image_data))))

    if filename == None:
        filename = NEXT_PLOT('png')

    image.save(filename)
    
# Write image data as a base64 string
def write_image_base64(image_data):
    image = Image.fromarray(numpy.flipud(numpy.rot90(normalise_greyscale(image_data))))

    buffer = io.BytesIO()

    image.save(buffer, format='PNG')

    buffer.seek(0)

    return base64.b64encode(buffer.getvalue()).decode('utf-8')

