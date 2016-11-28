from __future__ import print_function

import matplotlib
from matplotlib.cm import get_cmap

# For the colourbar
from mpl_toolkits.axes_grid1 import make_axes_locatable

# from matplotlib.offsetbox import AnnotationBbox, OffsetImage
# from matplotlib._png import read_png

from matplotlib import cm
import matplotlib.image as mpimg

import os
import numpy as np


# -----------------------------COLOURS EXPERIMENTAL----------------------------

# A function to load custom colormaps from .json files and
# use them in matplotlib :)
# From https://github.com/ccoughlin/ColormapCreator
# def load_colormap(json_file):
#     """Generates and returns a matplotlib colormap from the specified
#     JSON file,
#     or None if the file was invalid."""
#     colormap = None
#     with open(json_file, "r") as fidin:
#         cmap_dict = json.load(fidin)
#         if cmap_dict.get('colors', None) is None:
#             return colormap
#         colormap_type = cmap_dict.get('type', 'linear')
#         colormap_name = cmap_dict.get('name', os.path.basename(json_file))
#         if colormap_type == 'linear':
#             colormap = matplotlib.colors.LinearSegmentedColormap.\
#                         from_list(name=colormap_name,
#                         colors=cmap_dict['colors'])
#         elif colormap_type == 'list':
#             colormap = matplotlib.colors.ListedColormap(name=colormap_name,
#                                                         colors=cmap_dict['colors'])
#     return colormap

# Load custom palettes from the colours folder
# From https://github.com/benjaminaschultz/base16-ipython-matplotlibrc
# colorm = load_colormap('colours/monokai.json')
# cm = matplotlib.cm.get_cmap(colorm)

# -----------------------------------------------------------------------------

# Color palette for the plots --> This can be any colour palette from
# matplotlib
# cm = matplotlib.cm.get_cmap(name='Paired')

# My colours ------------------------------------------------------------------
# The palette_1 is the one currently being used. It is possible to specify
# any array of rgb colours and use them as a palette for the plots
#
# Convert hexadecimal colours to normalised  RGB. Code taken from stackoverflow
# http://stackoverflow.com/questions/214359/converting-hex-color-to-rgb-and-vice-versa


def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) / 255.
                 for i in range(0, lv, lv // 3))

# -----------------------------------------------------------------------------

# Colour palettes for the curves (used in cycle
d_palette2 = ['29376E', 'F52A00', 'F79400', '245723', 'DE2600']
d_palette1 = ['2C4C8F', 'FF5814', '88B459', 'D91D2C', '8F5536',
              '542437', '6F6F6F']

# Turn it into a palette readable by matplotlib
d_palette2 = [hex_to_rgb(c) for c in d_palette2]
d_palette1 = [hex_to_rgb(c) for c in d_palette1]


# -----------------------------------------------------------------------------


# Add the new MatLab default colourmap :)

# Get the path of the library (this file)
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)

# Remove the first two lines (comments)
# For the third column, remove the extra '\' (read until the sixth character)
# for this we use 'converters'
# Finally, do not consider the 0th column
my_rgb = np.loadtxt(dname + '/cmaps/parula.txt',
                    skiprows=2,
                    converters={3: lambda s: float(s[:6])},
                    usecols=(1, 2, 3))

parula_mpl = matplotlib.colors.ListedColormap(my_rgb, name='parula')
values = np.linspace(0., 1., len(my_rgb))
parula = get_cmap(parula_mpl)(values.copy())
parula[:, 0] = [int(x * 255) for x in parula[:, 0]]
parula[:, 1] = [int(x * 255) for x in parula[:, 1]]
parula[:, 2] = [int(x * 255) for x in parula[:, 2]]
parula[:, 3] = [int(x * 255) for x in parula[:, 3]]

parula = zip(parula[:, 0],
             parula[:, 1],
             parula[:, 2],
             parula[:, 3],)


def generate_linear_colormap(colour_list, cmap_name, stops=None):
    """
    colour_list     :: hex_list or list with rgb tuples
    """

    if not stops:
        stops = np.linspace(0, 1, len(colour_list))

    # We need the colour list in rgb
    if isinstance(colour_list[0], str):
        rgb_list = [hex_to_rgb(colour) for colour in colour_list]
    elif isinstance(colour_list[0], tuple):
        rgb_list = colour_list
    else:
        raise ValueError('Specify either a list with HEX colours as'
                         'strings or rgb (normalised) colours as tuples')

    colour_dict = {'red': (), 'green': (), 'blue': ()}

    for i, colour in enumerate(rgb_list):
        colour_dict['red'] += ((stops[i], colour[0], colour[0]),)
        colour_dict['green'] += ((stops[i], colour[1], colour[1]),)
        colour_dict['blue'] += ((stops[i], colour[2], colour[2]),)

    mpl_cmap = matplotlib.colors.LinearSegmentedColormap(cmap_name,
                                                         colour_dict,
                                                         256)

    # Reversed colourmap
    colour_dict_r = cm.revcmap(colour_dict)
    mpl_cmap_r = matplotlib.colors.LinearSegmentedColormap(cmap_name + '_r',
                                                           colour_dict_r,
                                                           256)

    return mpl_cmap, mpl_cmap_r


def generate_mayavi_cmap(mpl_cmap):
    """
    Generate a mayavi colourmap from a matplotlib colourmap
    """
    values = np.linspace(0., 1., 256)
    cmap = get_cmap(mpl_cmap)(values.copy())

    for i in range(4):
        cmap[:, i] = [int(x * 255) for x in cmap[:, i]]

    cmap = zip(cmap[:, 0], cmap[:, 1], cmap[:, 2], cmap[:, 3])

    return cmap


def plot_colormap(mpl_cmap):
    fig = matplotlib.pyplot.figure(figsize=(8, 3))
    ax = fig.add_axes([0.05, 0.80, 0.9, 0.15])

    cb = matplotlib.colorbar.ColorbarBase(ax, cmap=mpl_cmap,
                                          orientation='horizontal')

    cb.set_ticks([])
    cb.outline.set_visible(False)

# Custom Colormaps ------------------------------------------------------------

# Orange - Yellow - Blue ------------------------------------------------------
BuYlOr_rgbs = [(3 / 255., 33 / 255., 73 / 255.),
               (0.9976163, 0.9990772, 0.753402),
               (1, 69 / 255., 0.)
               ]

BuYlOr_mpl, BuYlOr_mpl_r = generate_linear_colormap(BuYlOr_rgbs, 'BuYlOr')
BuYlOr = generate_mayavi_cmap(BuYlOr_mpl)
BuYlOr_r = generate_mayavi_cmap(BuYlOr_mpl_r)

# Soft Rainbow ----------------------------------------------------------------
# From http://colorbrewer2.org/
softrbw_list = ['9E0142', 'D53E4F', 'F46D43', 'FDAE61', 'FEE08B', 'FFFFBF',
                'E6F598', 'ABDDA4', '66C2A5', '3288BD', '5E4FA2']

softrbw_mpl, softrbw_mpl_r = generate_linear_colormap(softrbw_list,
                                                      'softrbw')
softrbw = generate_mayavi_cmap(softrbw_mpl)
