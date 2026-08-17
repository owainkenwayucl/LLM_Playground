'''
   Tools for generating fractals.
   Owain Kenway, 2023, 2026
'''

import numpy
import os
import numba
import io
import base64

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

# generate a greyscale palette of colours for a given number of levels.
def generate_greyscale_palette(levels):
    palette = []
    for i in numpy.linspace(0,255,levels,dtype=int):
        shade = hex(i)[2:]
        if len(shade) == 1:
            shade='0' + shade
        colour = '#' + shade + shade + shade
        palette.append(colour)
    return palette    

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

def write_image_pillow(image_data, filename=None):
    from PIL import Image
    image_data = normalise_greyscale(image_data)
    image_data = numpy.flipud(numpy.rot90(image_data))
    image = Image.fromarray(image_data)

    if filename == None:
        filename = NEXT_PLOT('png')

    image.save(filename)
    
    
# Plot our image with matplotlib to a base64 encoded string
def write_image_matplotlib_base64(image_data, palette=None):
    import matplotlib.pyplot

    image = numpy.flipud(numpy.rot90(image_data[0]))

    buffer = io.BytesIO()
        
    if PRINT_MESSAGES:
        print('Writing to in memory buffer ...', end='', flush=True)
    matplotlib.pyplot.axis('off')
    if palette == None:
        matplotlib.pyplot.imshow(image)
    else:
        matplotlib.pyplot.imshow(image, cmap=palette)

    matplotlib.pyplot.savefig(buffer, bbox_inches='tight')
    
    buffer.seek(0)

    if PRINT_MESSAGES:
        print('done.')

    return base64.b64encode(buffer.getvalue()).decode('utf-8')

