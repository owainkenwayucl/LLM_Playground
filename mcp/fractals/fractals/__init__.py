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

# Show a tkinter window containing a given image.
def show_image(image_data, palette=None):
    import tkinter

    image = image_data[0]
    width = image.shape[0]
    height = image.shape[1]

    if (palette == None):
        palette = generate_greyscale_palette(image_data[1])

    window = tkinter.Tk()
    window.title("Fractal Image")
    canvas = tkinter.Canvas(window, width=width, height=height)
    canvas.pack(expand=tkinter.YES, fill=tkinter.BOTH) 
    for py in range(height):
        for px in range(width):
            canvas.create_oval(px,py,px+1,(py+1),fill=palette[image[px,py]], outline=palette[image[px,py]])            
    window.mainloop()

# Plot our image with matplotlib
def show_image_matplotlib(image_data, palette=None):
    import matplotlib.pyplot

    image = numpy.flipud(numpy.rot90(image_data[0]))

    matplotlib.pyplot.axis('off')
    if palette == None:
        matplotlib.pyplot.imshow(image)
    else:
        matplotlib.pyplot.imshow(image, cmap=palette)
    matplotlib.pyplot.show()

# Show image in terminal
def show_image_termshow(image_data):
    import fractals.termshow

    image = image_data[0]
    image_f = image.astype(numpy.float64)
    maximum = numpy.max(image)
    width = image.shape[0]
    height = image.shape[1]

    for j in range(height):
        for i in range(width):
            image_f[i,j] = image[i,j]/maximum

    fractals.termshow.show(image_f)

# Plot our image with matplotlib to a file
def write_image_matplotlib(image_data, palette=None, filename=None):
    import matplotlib.pyplot

    image = numpy.flipud(numpy.rot90(image_data[0]))

    if filename == None:
        filename = NEXT_PLOT('png')
        
    if PRINT_MESSAGES:
        print('Writing ' + filename + '...', end='', flush=True)
    matplotlib.pyplot.axis('off')
    if palette == None:
        matplotlib.pyplot.imshow(image)
    else:
        matplotlib.pyplot.imshow(image, cmap=palette)

    matplotlib.pyplot.savefig(filename, bbox_inches='tight')
    
    if PRINT_MESSAGES:
        print('done.')

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

# Dump image to PGM file
def write_image(image_data, palette=None, filename=None):
    image = image_data[0]

    if filename == None:
        filename = NEXT_PLOT('pgm')

    if PRINT_MESSAGES:
        print('Writing ' + filename + '...', end='', flush=True)

    width = image.shape[0]
    height = image.shape[1]

    depth = image_data[1]

    f = open(filename, 'w')

# Write PGM header
    f.write('P2\n')
    f.write('# Generated by https://github.com/owainkenwayucl/Fractals\n')
    f.write(str(width) + ' ' + str(height) + '\n')
    f.write(str(depth) + '\n')

# Write PGM
    for j in range(height):
        for i in range(width):
            f.write(str(image[i,j]) + ' ')
        f.write('\n')
        
    f.close()

    if PRINT_MESSAGES:
        print('done.')
