from __future__ import print_function

import warnings
imp_message = ("The Energy scale is in Joules. If you want to "
               "use a different energy scale, you can redefine the "
               "neb_plot_functions.scale variable and, to change the "
               "labels, use the neb_plot_functions.scale_label variable. "
               "In this way, the energy data will be divided by *scale*, "
               "e.g. \n "
               "import nebm_plot_tools as npt \n"
               "npt.scale = 1000 \n"
               "npt.scale_label = 'kJ' "
               )
warnings.warn(imp_message)

import matplotlib

# GLOBAL PARAMETERS
# Use LaTeX for the labels and numbering
# PDFLaTeX is for simplicity (Matplotlib uses XeTeX by default
# to handle characters, check the documentation in the webpage)


# Garamond Font settings !! ---------------------------------------------------

def set_matplotlib_Garamond():
    """
    We use here URW Garamond: https://www.ctan.org/pkg/urw-garamond?lang=en
    To install this font, we need
    this script: https://www.tug.org/fonts/getnonfreefonts/
    Download and run it with: texlua install-getnonfreefonts

    These settings use pdflatex
    """

    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = [
        # Set the roman
        r"\renewcommand{\rmdefault}{ugm}",
        # Set the seriff (I shot the Seriff)
        r"\renewcommand{\sfdefault}{ugm}",
        # Enconding according to the doc: <texmf>/doc/fonts/urw
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{textcomp}",
        # Set the math fonts using mathdesign
        r"\usepackage[urw-garamond]{mathdesign}",
        r"\usepackage{amsmath}"
    ]

    # Currently we set small fonts, but this is no good practice for journals:
    plt.rcParams.update({'font.size': 20})  # This should be equivalent
                                            # in structure than before
    plt.rcParams.update({'xtick.labelsize': 18})
    plt.rcParams.update({'ytick.labelsize': 18})

    # Font specifications
    # matplotlib.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Garamond']})

# -----------------------------------------------------------------------------

# Lato Font settings!! --------------------------------------------------------
# using these settings we loose the ability to use %matplotlib inline
# in a notebook!


"""
These settings use LuaLaTeX to render the plots, so we cannot
use the magic %matplotlib inline.

The fonts are set to Lato Light, which is sans, and most of the math fonts
try to use it by default through the math-unicode package. Other math fonts
are taken from cmbright

"""

def set_matplotlib_lua_Lato():
    plt.use('pgf')
    pgf_with_rc_fonts = {
        # use LuaLaTeX. Xetex has memory problems with large plots:
        "pgf.texsystem": "lualatex",
        "text.usetex": True,            # use LaTeX to write all text
        "font.family": "serif",         # use serif rather than sans-serif
        "font.serif": "Lato Light",     # use 'Lato' as the standard font
        "font.sans-serif": [],
        "font.monospace": "Fira Mono",  # use Ubuntu mono if we have mono
        "axes.labelsize": 26,           # LaTeX default is 10pt font.
        "font.size": 24,
        "legend.fontsize": 24,  # Make the legend/label fonts a little smaller
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "xtick.direction": 'out',
        "ytick.direction": 'out',
        "pgf.rcfonts": False,  # Use pgf.preamble, ignore standard mpl RC
        "text.latex.unicode": True,
        "pgf.preamble": [
            r'\usepackage{cmbright}',
            r'\usepackage{bm}',
            r'\usepackage{fontspec}',
            r'\setmainfont[BoldFont={Lato_700}, BoldItalicFont={Lato_700italic}]{Lato Light}',
            r'\setmonofont{Fira Mono}',
            r'\usepackage{unicode-math}',  # Math unicode chars with Lato font
            r'\setmathfont{Lato Light}'    # Math to Lato
        ]
    }

    plt.rcParams.update(pgf_with_rc_fonts)


def set_matplotlib_Lato():
    plt.rcParams.update({'font.size': 24})

    plt.rcParams.update({'xtick.labelsize': 24,
                         'ytick.labelsize': 24,
                         "legend.fontsize": 24,
                         "xtick.direction": 'out',
                         "ytick.direction": 'out',
                         'axes.labelsize': 26,
                         'axes.labelweight': 100,
                         'axes.formatter.use_mathtext': True
                         })

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Lato']
    plt.rcParams['font.cursive'] = ['Lato']
    plt.rcParams['font.weight'] = 100
    plt.rcParams["text.latex.unicode"] = True
    plt.rcParams["xtick.direction"] = 'out'
    plt.rcParams["ytick.direction"] = 'out'

    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.it'] = 'sans:italic'
    plt.rcParams['mathtext.bf'] = 'sans:bold:italic'
    plt.rcParams['mathtext.sf'] = 'sans'
    plt.rcParams['mathtext.default'] = 'it'
    plt.rcParams['mathtext.rm'] = 'sans'

# -----------------------------------------------------------------------------

matplotlib.rc('lines', lw=2)
# matplotlib.rcParams.update({'legend.fontsize': 13})

# Import matplotlib.pyplot after doing the changes
import matplotlib.pyplot as plt
# DPI for on screen figures
plt.rcParams['figure.dpi'] = 100

# Keep legends with the old Matplolib style (it changes in 2.0)
plt.rcParams['legend.fancybox'] = False
plt.rcParams['legend.framealpha'] = None
plt.rcParams['legend.scatterpoints'] = 3
plt.rcParams['legend.edgecolor'] = 'inherit'

# -----------------------------------------------------------------------------

# Colourmaps
from . import npt_colormaps as npt_cm

# To annotate Mayavi snapshots
from . import annotate_snapshots

# -----------------------------------------------------------------------------

import numpy as np
import itertools  # To iterate through markers
# import json  # For custom colormaps

import time
import sys
import os
import shutil
import subprocess
import re
from cycler import cycler

try:
    from mayavi import mlab
except:
    pass
from matplotlib.cm import get_cmap

# For the colourbar
from mpl_toolkits.axes_grid1 import make_axes_locatable

# from matplotlib.offsetbox import AnnotationBbox, OffsetImage
# from matplotlib._png import read_png

from matplotlib import cm
import matplotlib.image as mpimg

from matplotlib.widgets import Slider

# -----------------------------------------------------------------------------

# Utility for the functions that need to read npy files,
# (magnetization and skyrmion number)
#
# We need to sort the filenames from os.listdir() (which points to the
# npy files folder).
# In order to do this, we define a function that returns the
# integer in the file name, so Python can sort the names
# according to this criteria
def y(filen):
    """ Returns the integer in the filename 'filen'. For example,
    for 'image_13.npy', this function returns 13 as an integer
    """
    cname = ''
    for c in filen:
        if c.isdigit():
            cname += c
    return int(cname)

# Energy scaling
# This can be changed
# globally when importing, by doing: neb_plot_functions.scale = ()
# and
#
# By default we use J (wikipedia value)
SCALE = 1.
SCALE_LABEL = r'J'

# Plot Labels!
# DISTANCE_LABEL = r'$\sum\nolimits_j^i |\bm{Y}_{j+1} - \bm{Y}_{j}|$'
DISTANCE_LABEL = r'Distance from $\mathbf{Y}_{0}$ (a.u.)'

# This can be useful
kB = 1.3806488e-23
# scale_label = r'$k_B\, T$'

# Base Markers
MARKS = ['o', '^', 's', 'v', 'D', 'p', 'H', 'd', 'h']

# Default aspect ratio
# We can use the Golden Ratio (GR) for visual appealing
# If x = 8, then y = 8 / GR
# GR = 1.61803398875
GR = (np.sqrt(5.0) + 1.0) / 2.0
def_aspect = lambda size: (size, size / GR)

# Check is iPython is being used
try:
    from IPython.display import clear_output
    have_ipython = True
    # from IPython.display import HTML
    from IPython.display import Image
except ImportError:
    have_ipython = False

# COMMON FUNCTIONS (Labels, legends, etc)

# Default Colour Palette from the colourmpas library
D_PALETTE = npt_cm.d_palette1

# Default annotated numbers size
DEFAULT_NUM_FONTSIZE = 18

BACKEND = 'FIDIMAG'

"""

The base for the 3 plots styles we can generate from the NEB simulations data:

    energy , average magnetisation, skyrmion number

This class contains common functions for the 3 of them, using Matplotlib
functions

TODO: We can separate the main plot (with the decorations) to a different
abstract class, since other plot functions, such as the *plot_distances*, do
not use all of the available options, which can be confusing for the user. To
do this we need to understand better how multiple inheritance works

"""


class BaseNEBPlot(object):
    """

    Arguments:

    rel_folder           ::  Folder where the NEB simulation folders (vtks and
                             npys) and _energy.ndt and _dms.ndt files are.
                             By default, this value is the current folder where
                             the library is being called. If rel_folder is
                             specified, we avoid writing the full path to the
                             simulation files

    top_figure           ::  An optional  string with the relative path to the
                             overview image.  of the energy band, e.g
                             'png/simulation/grid.png' This will put the image
                             at the top of the main figure

    savefig              ::  A string with a filename (that includes a picture
                             file extension accepted by matplotlib) when a
                             PDF/png/jpeg/... version of the plot is required

    xlim                 ::  A python array with the corresponding limits in
                             case they want

    ylim                 ::  to be manually specified. For example: xlim=[0,
                             3]

                             Specify the 'ylim' if you get a strange huge
                             white space below the figure, since there is a
                             problem in matplotlib when the data point numbers
                             are out of range.

    num_labels           ::  A list with integers representing the curves
                             (starting from 0) that will be shown with
                             annotated numbers on every image. Default is zero
                             (the first curve on the 'sims' list)

    num_scale    ::  Optional image number annotation for the overview
                             plots.  This is passed as a list or integer:

                             A list for magnetisation plots:
                                 [num_factor_x, num_factor_y, num_factor_z]

                             A float for skyrmion number or energy plots

                             The image number of each data point are placed as
                             a factor of the difference between the smallest
                             and largest value in the y-axis.  A larger
                             num_scale would mean a larger separation
                             of the numbers from each data point. A negative
                             num_factor would put the numbers below the data
                             points.  If the factor is zero, the numbers are
                             not shown

                             If passed as a list, it needs to have the same
                             number of elements than num_labels. The numbers in
                             the n-th curve, whose index is in the num_labels
                             list, will be scaled by the n-th element in the
                             num_factor list. for example, if we want the 2nd
                             and 3rd curves to be annotated:
                                    num_labels = [2, 3]
                             and IF we want to scale the number positions of
                             both with the same factor, we use:
                                    num_factor =1.3
                             OR we can scale them by different factors doing:
                                    num_factor = [-1.1, 1.2]
                             which means that the 2nd curve numbers will be
                             scaled by a factor of -1.1 and the 3rd curve by a
                             factor of 1.2

    num_fontsize         ::  Annotated numbers size

    fig_aspect           ::  The ratio of aspect of the figure given as a
                             tuple.  If the values are larger, the figure size
                             increases.

    legend_position      ::  The legend position in the plot, given as a tuple
                             of (x, y) values. Coordinates x,y represent a
                             fraction of the width and height of the plot
                             respectively. The default value is (1.05, 1),
                             meaning outside the plot to the right (x=1.05),
                             and at the top of the figure (y=1).  The position
                             is from the upper left part of the box.  Instead
                             of a tuple it can also be passed a string with the
                             matplotlib options: 'upper left', 'lower right',
                             etc.

    legend_fontsize      ::  Legend fontsize (default 18)

    x_scale              ::  A power of ten (.. 0.1 , 10, 100, ...) to scale
                             the x axis

    images_top_align     ::  The bottom position of the figure that shows all
                             the images produced with mayavi in a grid.  This
                             figure is at the top of the plot. The value is
                             given as a factor of the plot height at the
                             bottom.  For example, top_left_align=0 would mean
                             that the top figure starts at the very energylower
                             region of the bottom plot. Default is 1.1

    nticks_{x,y}         ::  Number of ticks in the x or y axis

    energy_shift         ::  An image number that indicates the image used to
                             rescale the energy for every curve (energy band),
                             e.g. if energy_shift=0, for every line, the energy
                             will be shifted (subtracted) by the 0th image
                             energy. The y legend will be formatted to
                             'Rescaled Energy'


    x_scale              ::  A multiple of 10 to scale the x axis (dms) data

    magnetisation_plot   ::  [x_av, y_av, z_av]  --> list

    energy_plot          ::  [extrarg_list]  --> list

    num_labels           ::  An integer to plot numbers in the 'num_labels'th
                             curve of the 'sims' list

    num_scale_en ::  Option to scale the annotation numbers for the
                             optional energy axis in the magnetisation or sk
                             number plots

    num_fontsize         ::  Size of annotated numbers

    energy_shift         ::  An integer that indicates the image used to
                             rescale the energy for every curve (energy band),
                             e.g. if energy_shift=0, for every curve the energy
                             will be shifted (subtracted) by the 0th image
                             energy. The y legend will be formatted to
                             'Rescaled Energy'

    Legend and saving options for all the functions defined on this library.
    There is an optional argument 'optional_energy' which searches for an extra
    axis where the energy is defined

    optional_energy      ::  [ax2, ylim_en]

    legend_ncol          ::  number of columns for the legend when expanded on
                             top or at the bottom

    secondary_axis       ::  Add an extra axis to the right of the plot (only
                             useful for the Energy plot). This must be passed
                             as a list: [scale, scale_label] The 'scale' is to
                             divide the left axis scale by the specified
                             magnitude. Scale label is a string.

    legend_title         ::  A list: ['title string', (x, y)] where (x, y) is a
                             position. Set (0, 0) for the default centered
                             positioned title

    nticks_{x,y}         ::  Number of ticks


    Plot the skyrmion number, the average magnetisation OR the energy values in
    the 'dist-vs-sknum', 'dist-vs-m' or 'dist-vs-energy' plot functions,
    respectively.  This function aims to reduce the number of repeated code
    sections

    """

    def __init__(self,
                 sims,
                 **kwargs
                 ):

        accepted_kwargs = {'rel_folder': './',
                           # 'mesh': None,
                           # 'Ms': None,
                           'num_labels': [0],
                           'num_scale': 0.04,
                           'num_scale_en': None,
                           'ylim_en': None,
                           'x_scale': None,
                           'num_fontsize': DEFAULT_NUM_FONTSIZE,
                           'energy_shift': None,
                           'scale': SCALE,
                           'scale_label': SCALE_LABEL,
                           'fig_aspect': def_aspect(7.),
                           'markers': MARKS,
                           'xlim': None,
                           'ylim': None,
                           'top_figure': None,
                           'savefig': None,
                           'interpolate_energy': False,
                           # Arguments for top and bottom images plots --------
                           'cmap': 'RdYlBu',
                           # Left alignment for both top and bottom figures:
                           'top_figure_h_shift': 0,
                           'top_figure_v_shift': 0,
                           'top_figure': None,
                           'bottom_figure': None,
                           'bottom_figure_h_shift': 0,
                           'bottom_figure_v_shift': 0,
                           'top_figure_frame': None,
                           'bottom_figure_frame': None,
                           'colorbar': 'top',
                           'colorbar_label': r'$m_{z}$',
                           'colorbar_offset': 10,
                           # Decorations --------------------------------------
                           'optional_energy': False,
                           'grid': True,
                           'x_scale': None,
                           'legend_ncol': 2,
                           'secondary_axis': None,
                           'legend_title': None,
                           'nticks_x': None,
                           'nticks_y': None,
                           'legend_position': 'upper right',
                           'legend_fontsize': 18
                           }

        for (option, key) in accepted_kwargs.items():
            setattr(self, option, kwargs.get(option, key))

        self.sims = sims
        self.ax2 = None

        self.plots = []
        self.labels = []

        if self.x_scale:
            if not np.log10(self.x_scale).is_integer():
                raise ValueError('Use a power of 10 to scale the axis')

        self.init_figure()

    def generate_x_data_from_dms(self, names):
        # Load the distances (btwn images) files associated to this
        # simulation, from the dms file, for an specific step.  Scale the
        # axis if necessary
        dms = np.loadtxt(self.rel_folder + names[self.dms_index])[names[self.step_index]][1:]
        if self.x_scale:
            dms *= self.x_scale

        # Compute the total distance of a point from one of the extremes (the
        # extremes are energy minima). It is only necessary to sum the
        # distances up an specific point
        x_data = [0]
        for i in range(len(dms)):
            x_data.append(np.sum(dms[:i + 1]))

        return x_data

    def generate_npys_file_list(self, names):
        # List all the npy files of the required NEB simulation from the
        # folder that starts with the simulation name inside 'npys'
        npys_flist = os.listdir('{}npys/{}_{}'.format(
            self.rel_folder,
            names[self.sim_index],
            names[self.step_index]
            )
            )
        # Now sort the filenames with the fuction specified before
        npys_flist.sort(key=y)

        return npys_flist

    def init_figure(self):
        """
        Generate main figure and axes. Also initiates the colour cycle
        for the curves
        """
        # Initiate markers iterator
        self.markers = itertools.cycle(MARKS)

        self.fig = plt.figure(figsize=self.fig_aspect)
        if not self.top_figure:
            # Set the image a bit larger
            # self.fig.set_size_inches(self.fig_aspect[0] * 1.15,
            #                          self.fig_aspect[1] * 1.15
            #                          )
            self.ax = self.fig.add_subplot(111)
        else:
            # self.ax = self.fig.add_axes([0.1, 0.1, 1, 1])
            # These are the default values from matplotlib figures:
            self.ax = self.fig.add_axes([0.125, 0.125, 0.9 - 0.125, 0.9 - 0.125])

        # ax.set_color_cycle([cm(k) for k in np.linspace(0, 1, len(sims) * 3)])
        self.ax.set_prop_cycle(cycler('color', D_PALETTE))

    def init_secondary_axis(self):
        """
        A secondary axis to the right side of the plot,
        mainly used to plot energy data

        zorder is necessary because somehow, the curves dissapear
        by default (BUG?)
        """
        self.ax2 = self.ax.twinx()
        self.ax2.set_zorder(1)
        self.secondary_axis_label = ''

    def annotate_numbers(self, ax,
                         x_data,
                         y_data,
                         ylim=None,
                         scale=None,
                         color=None,
                         ):
        """
        Annotate numbers for every data point (NEB images) for the
        specified axis. ylim limits are necessary because the figure
        behaves strangely when an annotated point is outside the
        plot region

            y_data  :: Must be a Numpy array!

            color  (Experimental)

        """

        # Annotate numbers below points if specified. The num_scale is
        # set as the scale factor in the plotting functions
        if scale:
            # Draw only the points inside the ylim range if corresponds. There
            # is a problem with the white space when they are out of range
            nums = np.arange(len(x_data))

            # ax_scale = np.max(np.abs(y_data[1:] - y_data[:-1]))
            # ax_scale = ax_scale * scale

            # Draw only the points inside the ylim range if corresponds
            # There is a problem with the white space when they are
            # out of range
            if ylim:
                nums = nums[y_data < ylim[1]]

            if not color:
                color = 'black'
            for i in nums:
                a = ax.annotate(str(i),
                                xy=(x_data[i], y_data[i]),
                                textcoords='data',
                                horizontalalignment='center',
                                fontsize=self.num_fontsize,
                                color=color
                                )
                # Get annotation position as display data
                a_d = self.ax.transData.transform(a.get_position())
                # Shift this position by adding a scale in Axes-coordinates
                # (from 0 to 1) transformed in display data
                new_a = a_d + (self.ax.transAxes.transform((0, scale))
                               - self.ax.transAxes.transform((0, 0)))
                # Get the new position back into data coordinates and update
                # the annotation position
                a.set_position(self.ax.transData.inverted().transform(new_a))

                # This might not work perfectly since, when adding new curves,
                # self.ax changes its limits, so larger scales are required
                # for the last curves

    # -----------------------------------------------------------------------

    def insert_colorbar(self, ax):
        # COLORBAR

        # Define the colormap from the figure
        cmap = matplotlib.cm.get_cmap(name=self.cmap)
        norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)

        # Add axes for the colorbar with respect to the top image
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.03)

        # Colorbar
        cbar = matplotlib.colorbar.ColorbarBase(cax,
                                                cmap=cmap,
                                                norm=norm,
                                                ticks=[-1, 0, 1],
                                                orientation='vertical',
                                                )
        cbar.set_label(self.colorbar_label, rotation=270,
                       labelpad=self.colorbar_offset)

    def insert_top_plot(self):

        # Get the positions of the axes box (bottom left and top right corners):
        #   [[x0, y0], [x1, y1]]
        ax_pos = self.ax.get_position()

        # ---------------------------------------------------------------------
        # Load the image in png format
        # im_grid = read_png(grid_filen)
        im_grid = mpimg.imread(self.top_figure)
        h, w = float(im_grid.shape[0]), float(im_grid.shape[1])

        # The height of the top figure will be given as a proportion of the
        # height of the main figure. The main figure height H is 'y1 - y0' in
        # relative coordinates
        images_height = (self.fig_aspect[0] * h / w) / self.fig_aspect[1]
        # images_width = 1

        # Add axis for an image above the plot
        # [left, bottom, width, height]
        # If bottom = ax_pos.y1, it works for every fig_aspect, and there is
        # a small gap between the main figure and the top plot
        # We will try to reduce the gap a little, but this doesn't work
        # when height > width for the main figure
        self.top_ax = self.fig.add_axes([ax_pos.x0 + self.top_figure_h_shift,
                                         ax_pos.y1 - 0.05 + self.top_figure_v_shift,
                                         ax_pos.x1 - ax_pos.x0,
                                         images_height
                                         ],
                                        # aspect=900/2100
                                        )

        self.top_ax.set_xticks([])
        self.top_ax.set_yticks([])

        # ---------------------------------------------------------------------

        # This will fill up the second axis, top_ax
        self.top_ax.imshow(im_grid,
                           # extent=[0, im_grid.shape[1], 0, im_grid.shape[0]]
                           )

        if self.top_figure_frame:
            for spine in self.top_ax.spines.values():
                spine.set_edgecolor(self.top_figure_frame)

    def insert_bottom_plot(self):
        # Get the positions of the axes box (bottom left and top right corners):
        #   [[x0, y0], [x1, y1]]
        ax_pos = self.ax.get_position()

        # An extra figure will be plotted down the plot (EXPERIMENTAL)
        im_grid = mpimg.imread(self.bottom_figure)
        h, w = float(im_grid.shape[0]), float(im_grid.shape[1])
        images_height = (self.fig_aspect[0] * h / w) / self.fig_aspect[1]
        images_width = 1

        # Add axis for an image above the plot
        # [left, bottom, width, height]
        self.bottom_ax = self.fig.add_axes([ax_pos.x0 + self.bottom_figure_h_shift,
                                            -ax_pos.y1 + 0.05 + self.bottom_figure_v_shift,
                                            ax_pos.x1 - ax_pos.x0,
                                            images_height
                                            ],
                                           )

        self.bottom_ax.set_xticks([])
        self.bottom_ax.set_yticks([])

        self.bottom_ax.imshow(im_grid)

        if self.bottom_figure_frame:
            for spine in self.bottom_ax.spines.values():
                spine.set_edgecolor(self.bottom_figure_frame)

    def insert_images_plots(self):
        if self.top_figure:
            self.insert_top_plot()
            if self.colorbar == 'top':
                self.insert_colorbar(self.top_ax)

        if self.bottom_figure:
            self.insert_bottom_plot()
            if self.colorbar == 'bottom':
                self.insert_colorbar(self.top_ax)

    def set_limits_and_ticks(self):
        """
        We separate this function from the decorate_plot to set up a secondary
        axis after the limits and ticks have changed (in the energy plot the
        axes are linked)
        """

        # Change limits if specified
        if self.xlim:
            self.ax.set_xlim(self.xlim)
        if self.ylim:
            self.ax.set_ylim(self.ylim)

        # Set limits for the energy axis
        # if self.optional_energy and self.optional_energy[0]:
        #     self.optional_energy[0].set_ylim(self.optional_energy[1])

        if self.nticks_x:
            self.ax.locator_params(axis='x', nbins=self.nticks_x)
        if self.nticks_y:
            self.ax.locator_params(axis='y', nbins=self.nticks_y)

    # Plot decorations --------------------------------------------------------
    def decorate_plot(self,
                      xlabel,
                      ylabel,
                      grid=True
                      ):
        # Make axes level to top
        self.ax.set_zorder(1)

        if grid:
            self.ax.grid()

        # Legend Specifications
        # Change configuration if ax2 exists
        # (this is for the right axis of different plots that have
        # the energy curve as an optional value)

        if self.legend_position:
            try:
                self.optional_energy[0].set_ylabel(r'Energy  [ ' + self.scale_label + r' ]')
                self.optional_energy[0].tick_params(axis='both', which='major')

                if not isinstance(self.legend_position, str):
                    leg = self.ax.legend(self.plots, self.labels,
                                         loc=2,
                                         bbox_to_anchor=self.legend_position,
                                         prop={'size': self.legend_fontsize})
                else:
                    if self.legend_position == 'expand top':
                        leg = self.ax.legend(self.plots, self.labels,
                                             ncol=self.legend_ncol, loc='lower left',
                                             bbox_to_anchor=(0, 1.02, 1, 1),
                                             mode='expand', borderaxespad=0.,
                                             fontsize=18
                                             )

                    elif self.legend_position == 'expand bottom':
                        leg = self.ax.legend(self.plots, self.labels,
                                             ncol=self.legend_ncol, loc='upper left',
                                             bbox_to_anchor=(0, 1, 1, 1),
                                             mode='expand', borderaxespad=0.,
                                             fontsize=18
                                             )
                    else:
                        leg = self.ax.legend(loc=self.legend_position,
                                             # borderaxespad=0.,
                                             prop={'size': self.legend_fontsize}
                                             )

            # Otherwise, Set the legend as usual
            except:
                # Legend specifications
                # loc together with bbox_to_anchor will make the
                # upper left (that is '2') part of the legend box to be located
                # at the bbox_to_anchor coordinate
                # bbox_to_anchor can be a 4-tuple: (x,y,width,height)
                if isinstance(self.legend_position, str):
                    if self.legend_position == 'expand top':
                        leg = self.ax.legend(self.plots, self.labels,
                                             ncol=self.legend_ncol, loc='lower left',
                                             bbox_to_anchor=(0, 1.02, 1, 1),
                                             mode='expand', borderaxespad=0.,
                                             prop={'size': self.legend_fontsize}
                                             )
                    elif self.legend_position == 'expand bottom':
                        leg = self.ax.legend(self.plots, self.labels,
                                             ncol=self.legend_ncol, loc='upper left',
                                             bbox_to_anchor=(0, -1.22, 1, 1),
                                             mode='expand', borderaxespad=0.,
                                             prop={'size': self.legend_fontsize}
                                             )
                    else:
                        leg = self.ax.legend(self.plots, self.labels,
                                             loc=self.legend_position,
                                             prop={'size': self.legend_fontsize})
                else:
                        leg = self.ax.legend(self.plots, self.labels,
                                             bbox_to_anchor=self.legend_position,
                                             loc=2,
                                             # this option is the spacing between
                                             # the axes and the legend box:
                                             # borderaxespad=0.,
                                             prop={'size': self.legend_fontsize})
            if self.legend_title:
                leg.set_title(self.legend_title[0])
                leg.get_title().set_position(self.legend_title[1])
                leg.get_title().set_fontsize(self.legend_fontsize)

            # Legend on top ??
            leg.set_zorder(10)

        # Update the x axis label if the scaling option is True
        # The Axis is scaled with the 1 / x_scale factor (hence the
        # minus sign). We extract log10 to get the number of zeros
        if self.x_scale and not xlabel.endswith('(a.u.)'):
            xlabel = (xlabel + ' '
                      + r'$\times 10^{'
                      + '{}'.format(-(int(np.log10(self.x_scale))))
                      + r'}$'
                      )

        # Update labels
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)

        # Larger fonts for the ticks labels
        # plt.tick_params(axis='both', which='major', labelsize=13)

        # Try to avoid labels to be cut off from the figure
        # fig.tight_layout()

    def generate_secondary_axis_energy_scale(self):
        """

        secondary_axis  :: This argument will be initialised in the
                           corresponding children classes and will be passed
                           as a 2-element list, where the 1st element is
                           the scale (float) and the second element is a
                           label (str)

        """
        self.init_secondary_axis()
        y1, y2 = self.ax.get_ylim()
        self.ax2.set_ylim(y1, y2)
        ticks2 = matplotlib.ticker.FuncFormatter(
            lambda x, pos: '{:.2f}'.format(x / self.secondary_axis[0]))
        self.ax2.yaxis.set_major_formatter(ticks2)
        self.ax2.set_ylabel(self.secondary_axis[1])

    def secondary_axis_energy_plot(self, x_data, names):

        energy_data = np.loadtxt(self.rel_folder +
                                 names[self.energy_index])[names[self.step_index]][1:]
        if self.interpolate_energy:
            x_interp, y_interp = self.interpolate_energy_curve(
                '{}npys/{}_{}'.format(self.rel_folder,
                                      names[self.sim_index],
                                      names[self.step_index],
                                      ))
            y_interp /= self.scale

        # Scale energy
        y_data = energy_data / self.scale

        if self.energy_shift is not None:
            y_shift = y_data[self.energy_shift]
            # From the energy of every image in the band, subtract
            # the energy of the image specified by 'energy_shift'
            y_data -= y_shift

            # Redefine zero point of interpolated data if corresponds
            if self.interpolate_energy:
                y_interp -= y_shift

        p2, = self.ax2.plot(x_data, y_data,
                            marker=next(self.markers),
                            color=next(self.ax._get_lines.prop_cycler)['color'],
                            markeredgecolor='black',
                            markeredgewidth=1.5,
                            lw=2 * int(not bool(self.interpolate_energy)),
                            )

        self.plots.append(p2)
        self.labels.append(r'Energy')

        if self.interpolate_energy:
            p2.set_zorder(3)
            self.ax2.plot(x_interp, y_interp, lw=2,
                          color=p2.get_color(), zorder=2)

        self.annotate_numbers(self.ax2, x_data, energy_data / self.scale,
                              scale=self.num_scale_en,
                              ylim=self.ylim_en
                              )

    def save_plot(self):
        # Save if specified
        if self.savefig:
            plt.savefig(self.savefig, bbox_inches='tight')

    def interpolate_energy_curve(self, npys_folder):
        """

        This function loads the images from the specified NPY folder into a
        NEBM simulation and returns two arrays with the data to interpolate an
        energy curve. Thus, the function requires that: self.nebm_class,
        self.simulation and self.interp_res are defined (how they are defined
        depends on the child class).

        """
        images = [np.load(os.path.join(npys_folder, _file))
                  for _file in sorted(os.listdir(npys_folder),
                                      key=lambda f: int(re.search('\d+', f).group(0))
                                      )]
        nebm_sim = self.nebm_class(self.simulation, images)
        l, E = nebm_sim.compute_polynomial_approximation(self.interp_res)

        return l, E


# #############################################################################
# #################  MAYAVI2 VISUALIZATIONS   #################################
# #############################################################################

def plot_mayavi2(simname,
                 ran_images,
                 cmap='hot', reversed_map=False,
                 component='z-component',
                 savef=None, gridn=None,
                 camera_elevation=0,
                 camera_azimuth=0,
                 camera_distance=None,
                 zoom=1,
                 thumbnails_size='300x300',
                 rel_folder='./',
                 # extension='vtu',
                 # vtk_name='image_',
                 interpolation='gouraud',
                 # fidimag=False,
                 text_fontsize=170,
                 text_color='black',
                 ):
    """
    Return a sequence of images showing the z component of the magnetization
    of the files produced with the NEB algorithm.
    It is assumed that the directory with the NEB vtk files is 'vtks/'
    (from the NEB simulation outputs). The files for different simulations are
    distinguished by the simulation name (with the simulation step)

    simname         --> String with the NEB simulation VTK folder
                       (e.g. 'neb_k1e5_400' or 'neb_k1e5_1600st_0')

    ran_images      --> an array of type 'range' with the step number of the
                         images to be plotted (images start from 0 to maxsteps)
    cmap            --> A colormap accepted by Mayavi2
    reversed_map    --> Set True if you want to reverse the colour map (cmap)
    component       --> A string with the component of the magnetisation to
                        be plotted as a surface colourmap.
                        This name is supplied to Mayavi as:
                        'i-component' where i = x,y,z. By default, this
                        option is: 'z-component'
    savef           --> Filename with an extension accepted by the 'montage'
                        command from bash. This option will merge all the
                        images in a single file where the images are
                        structured in a grid.
    gridn           --> The structure of the grid for the 'montage' command.
                        For instance, gridn='6x3' will produce a grid with
                        6 columns and 3 rows of images
    camera_elevation -> The option in mayavi2 to set the view with a polar
                        angle. This values goes from 0 to 180 (degrees). Zero
                        means view from top (spherical coords).

    azimuth          -> From Mayavi2 documentation: The azimuthal angle (in
                        degrees, 0-360), i.e. the angle subtended by
                        the position vector on a sphere projected on to the
                        x-y plane with the x-axis.

    rel_folder --> The relative path of the 'vtks' folder can be changed
                   specifying its parent folder as: 'folder/'
                   (so we have 'folder/npys/')

    extension (DEPRECATED) --> The VTK files extension. By default it is vtu, but it
                   could also change, to 'vtk' or 'pvd' for example

    interpolation --> Surface interpolation: 'gouraud', 'phong', 'flat'

    """

    # Figure specs
    f = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(200, 200))

    try:
        if BACKEND == 'FINMAG':
            # Read a data file.
            data = mlab.pipeline.open('{}vtks/{}/'
                                      'image_000000.vtu'.format(rel_folder,
                                                                simname,
                                                                ))
        elif BACKEND == 'FIDIMAG':
            # Read a data file.
            data = mlab.pipeline.open('{}vtks/{}/'
                                      'image_000000.{}'.format(rel_folder,
                                                               simname,
                                                               'vtk'  # can be an option
                                                               ))
    except:
        print('Not a valid VTK file (check the files path)')
        sys.exit(1)

    # Currently fidimag does not filter the points with zero Ms
    # We will avoid them using a threshold value for the magnetisation norm
    # Finmag does not have this problem
    if BACKEND == 'FINMAG':
        # Extract vector components.
        vecomp = mlab.pipeline.extract_vector_components(data)
    else:
        # Extract vector norm and filter the points whose norm is
        # zero (we can set an arbitrary low value)
        vnorm = mlab.pipeline.extract_vector_norm(data)

        vtres = mlab.pipeline.threshold(vnorm)
        try:
            vtres.lower_threshold = 0.01
        except:
            print('No points with zero vector norm')

        # Extract vec comp and plot
        vecomp = mlab.pipeline.extract_vector_components(vtres)

    # Extract z-component of the data
    vecomp.component = component

    # Plot the surface with a colurmap according to the mz component
    # If the cmap is not a string, pass it as a table (document this)
    if isinstance(cmap, str):
        surf = mlab.pipeline.surface(vecomp, vmax=1, vmin=-1, colormap=cmap)
    else:
        surf = mlab.pipeline.surface(vecomp, vmax=1, vmin=-1)
        surf.module_manager.scalar_lut_manager.lut.table = cmap

    surf.actor.property.interpolation = interpolation

    if reversed_map:
        surf.module_manager.scalar_lut_manager.reverse_lut = True

    # View from top as default
    mlab.view(elevation=camera_elevation,
              azimuth=camera_azimuth,
              distance=camera_distance
              )
    # f.scene.z_plus_view()

    # Insert a text with the image number
    # t = mlab.text(text_position[0], text_position[1],
    #               '00', width=0.22,
    #               line_width=1.,
    #               color=text_color
    #               )

    for step in ran_images:
        # Change the numeration of the files in the Mayavi input
        data.timestep = step
        # We need to update the threshold value in case we use
        # finmag's vtk files
        if BACKEND == 'FIDIMAG':
            try:
                # If there is no point with zero 'm', this threshold fails
                vtres.lower_threshold = 0.01
            except:
                pass
        # Update image label
        # t.text = str(step).zfill(2)  # rjust() --> Fill with spaces
        # Zoom for saving purposes
        f.scene.magnification = zoom
        # Save in an appropriate file
        f.scene.save('{}_image_{}.png'.format(simname, str(step).zfill(2)))
    # To avoid Mayavi to crash from iPython (if run in script, it
    # should be okay without show() )
    mlab.show()

    # Save files ----------------------------------------------------------
    imrootd = os.path.join('png', '{}'.format(simname))
    if not os.path.exists(imrootd):
        os.makedirs(imrootd)
    else:
        shutil.rmtree(imrootd)
        os.makedirs(imrootd)

    files = os.listdir('.')
    for f in files:
        if f.startswith(simname) and f.endswith('png') and os.path.isfile(f):
            shutil.move(f, imrootd)

    # Annotate snapshots with Matplotlib
    annotate_snapshots.annotate_snapshots(imrootd,
                                          color=text_color,
                                          fontsize=text_fontsize
                                          )

    # Save a grid of images if savef is True
    if savef:
        subprocess.Popen('cd {} && montage -geometry {} -tile {} {}* {}'.
                         format(imrootd, thumbnails_size, gridn,
                                simname, savef), shell=True)


def plot_mayavi2_old(simname, simstep, maxsteps, save_every_step, imnum,
                     ran_images, cmap='hot', savef=None, gridn=None,
                     camera_elevation=0, zoom=1, thumbnails_size='300x300'):
    """
    (The newest version of finmag and the NEB code, save the vtk
    files in different folders. This is much more simple and tidy.
    This old version is only for plotting all the files generated
    with the previous code, which are saved in a common folder)

    Return a sequence of images showing the z component of the magnetization
    of the files produced with the NEB algorithm.
    It is assumed that the directory with the NEB vtk files is 'vtks/'
    (from the NEB simulation outputs). The files for different simulations are
    distinguished by the simulation name

    simname         --> String with the NEB simulation name (e.g. 'neb_k1e5'
                        or 'neb_k1e5_1600st')
    simstep         --> Step to be processed with this function. It is assumed
                        that the files for this particular step of the NEB
                        algorithm exist in the vtks/ directory. In a different
                        case, Mayavi2 will exit with an error.
    maxsteps        --> Maximum number of steps specified in the NEB algorithm.
                        If the simulation stopped before, this is the total
                        number of steps computed.
    save_every_step --> The number specified in the NEB simulation to save the
                        vtk file severy certain number of steps.
    imnum           --> Number of images used in the NEB algorithm.
    ran_images      --> an array of type 'range' with the step number of the
                         images to be plotted (images start from 0 to maxsteps)
    cmap            --> A colormap accepted by Mayavi2
    savef           --> Filename with an extension accepted by the 'montage'
                        command from bash. This option will merge all the
                        images in a single file where the images are
                        structured in a grid.
    gridn           --> The structure of the grid for the 'montage' command.
                        For instance, gridn='6x3' will produce a grid with
                        6 columns and 3 rows of images
    camera_elevation -> The option in mayavi2 to set the view with a polar
                        angle. This values goes from 0 to 180 (degrees). Zero
                         means view from top (spherical coords).

    """

    # Figure specs
    f = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(200, 200))

    # Read a data file.
    data = mlab.pipeline.open('vtks/{}_{}000000.vtu'.format(simname, simstep))

    # Extract vec comp and plot
    vecomp = mlab.pipeline.extract_vector_components(data)
    vecomp.component = 'z-component'  # Extract z-component of the data
    # If the cmap is not a string, pass it as a table (document this)
    if isinstance(cmap, str):
        surf = mlab.pipeline.surface(vecomp, vmax=1, vmin=-1, colormap=cmap)
    else:
        surf = mlab.pipeline.surface(vecomp, vmax=1, vmin=-1)
        surf.module_manager.scalar_lut_manager.lut.table = cmap

    mlab.view(elevation=camera_elevation)  # View from top
    t = mlab.text(0.8, 0.9, '00',  width=0.16, line_width=1., color=(0, 0, 0))

    # The newest version of finmag starts at 0, so we could eliminate the
    # == 1  option in the future
    if simstep == 1 or simstep == 0:
        initstep = 0
    elif maxsteps <= save_every_step:
        initstep = imnum
    else:
        div = simstep % save_every_step
        initstep = (int((simstep - div) / save_every_step)) * imnum

    for step in ran_images:
        # Change the numeration of the files
        data.timestep = initstep + step
        t.text = str(step).zfill(2)  # rjust() --> Fill with spaces
        f.scene.magnification = zoom
        f.scene.save('{}_image_{}.png'.format(simname, str(step).zfill(2)))
    mlab.show()

    # Save files ----------------------------------------------------------
    imrootd = os.path.join('png', '{}_{}'.format(simname, simstep))
    if not os.path.exists(imrootd):
        os.makedirs(imrootd)
    else:
        shutil.rmtree(imrootd)
        os.makedirs(imrootd)
    files = os.listdir('.')
    for f in files:
        if f.startswith(simname) and f.endswith('png') and os.path.isfile(f):
            shutil.move(f, imrootd)

    if savef:
        subprocess.Popen('cd {} && montage -geometry {} -tile {} {}* {}'.
                         format(imrootd, thumbnails_size, gridn,
                                simname, savef), shell=True)


# -----------------------------------------------------------------------------
# GENERAL PLOTS ---------------------------------------------------------------
# -----------------------------------------------------------------------------

class plot_dist_vs_energy(BaseNEBPlot):
    """

    sims  -->  An array of arrays with 4 elements each:
               The array has the structure

               [[energy_file, dms_file, label, step], [%], ...]

               * The dms_file can be specified by the whole route to the
                 file, e.g. 'data/simulation/{simname}_dms.ndt'

    secondary_axis  :: A secondary axis for the energy

    widget          ::

    interpolate_energy   ::  Specify a list with three elements: a Fidimag NEBM
                             simulation class, a simulation object and an
                             integer indicating the number of interpolated
                             points; in order to compute a smooth approximation
                             of the energy bands, using the information from
                             the nebm vectors.  In the sims list, the last
                             argument must be the path towards the NPY file to
                             interpolate the data

    """
    # Inherit the doc from the BasePlot
    __doc__ += BaseNEBPlot.__doc__

    def __init__(self, *args, **kwargs):

        setattr(self, 'widget', kwargs.get('widget', None))

        super(plot_dist_vs_energy, self).__init__(*args,
                                                  **kwargs)

        (self.energy_index,
         self.dms_index,
         self.label_index,
         self.step_index,
         ) = (0, 1, 2, 3)

        if self.interpolate_energy:
            self.nebm_class = self.interpolate_energy[0]
            self.simulation = self.interpolate_energy[1]
            self.interp_res = self.interpolate_energy[2]
            self.nebm_index = 4

        self.plot_neb_curves()

        # Decorate Plots ------------------------------------------------------

        # OPTIONAL Initial State Energy Plots
        # self.plot_initial_energy_curve()

        # Legend decorations and saving options
        # if self.energy_shift is not None:
        #     self.energy_label = r'Rescaled Energy  (' + self.scale_label + r')'
        # else:
        self.energy_label = r'Energy  (' + self.scale_label + r')'

        # Top and bottom plots ------------------------------------------------
        self.insert_images_plots()

        # Limits
        self.set_limits_and_ticks()

        # Secondary axis
        if self.secondary_axis:
            self.generate_secondary_axis_energy_scale()

        # Decorations ---------------------------------------------------------
        self.decorate_plot(DISTANCE_LABEL, self.energy_label)

        # Redefine the legend styles in case we use interpolated lines (we
        # separated the plots into markers and curve). Only do this if the
        # legend exists (in case None was used for legend_position)
        if self.interpolate_energy and self.ax.get_legend():
            for mark in self.ax.get_legend().legendHandles:
                mark.set_linestyle('-')
                mark.set_linewidth(2)
                mark.lineStyles['-'] = '_draw_solid'

        # Optional widget -----------------------------------------------------
        if self.widget:
            self.plot_widget()
            self.step.on_changed(self.update)
            plt.show()

        #
        self.save_plot()

    def plot_neb_curves(self):

        for names in self.sims:

            x_data = self.generate_x_data_from_dms(names)

            # Load the energies in 'data'
            y_data = np.loadtxt(self.rel_folder +
                                names[self.energy_index]
                                )[names[self.step_index]][1:]

            # Get the data from interpolation if the option was specified
            if self.interpolate_energy:
                x_interp, y_interp = self.interpolate_energy_curve(
                    self.rel_folder + names[self.nebm_index])

            if self.energy_shift is not None:
                y_shift = y_data[self.energy_shift]
                # From the energy of every image in the band, subtract
                # the energy of the image specified by 'energy_shift'
                y_data -= y_shift

                # Redefine zero point of interpolated data if corresponds
                if self.interpolate_energy:
                    y_interp -= y_shift

            # If self.interpolate_energy is specified, the int(bool(..)) will
            # return 1, otherwise it will give 0
            p = self.plot_energy_curve(
                x_data, y_data, names,
                lw=2 * int(not bool(self.interpolate_energy)),
                marker=next(self.markers))

            if self.interpolate_energy:
                p.set_zorder(3)
                self.ax.plot(x_interp, y_interp / self.scale, lw=2,
                             color=p.get_color(), zorder=2)

    def plot_energy_curve(self, x_data, y_data, names,
                          lw=2, marker=None):
        # Plot and save the plot element to use it in the legend options
        p, = self.ax.plot(x_data, y_data / self.scale,
                          marker=marker,
                          lw=lw,
                          # markerfacecolor=linec.get_color(),
                          markeredgecolor='black',
                          markeredgewidth=1.5,
                          # label=names[2]
                          )

        self.plots.append(p)
        self.labels.append(names[self.label_index])

        # Number for each data point ------------------------------------------

        # ONLY for the n-th curve, IF n IS in the num_labels list (this way the
        # numbers are not overlapped)

        # If number annotation is a list, we will scale the position according
        # to the factors in num_scale. For this, we find the index of
        # the curve in the 'sims' list, and check which index corresponds to
        # this curve in the num_labels list Therefore, num_labels must be equal
        # in length than num_scale
        # For example, if I have
        # sims = [[energy_file1, dms_file1, 'string1', num1],
        #         [energy_file2, dms_file2, 'string2', num2]
        #        ]
        # num_labels = [1]   --> only the sims[1] curve will be annotated
        # num_scale = [0, 1]
        #
        # Then, [energy_file2, ..] has the 1th index in the 'sims' list,
        # and THIS element IS the 0th index in the num_labels list
        # Thus, we use the 0th element of num_factor to scale the 1th curve
        #
        if self.num_labels and self.sims.index(names) in self.num_labels:
            if isinstance(self.num_scale, list):
                num_index = self.num_labels.index(self.sims.index(names))
                scale = self.num_scale[num_index]
            else:
                scale = self.num_scale

            self.annotate_numbers(self.ax, x_data, y_data / self.scale,
                                  ylim=self.ylim,
                                  scale=scale
                                  )

        # ---------------------------------------------------------------------

        return p

    def plot_widget(self):
        max_steps = len(np.loadtxt(self.sims[0][0]))
        axcolor = 'lightgoldenrodyellow'
        ax_slider = plt.axes([0.13, 0.95, 0.77, 0.03], facecolor=axcolor)
        self.step = Slider(ax_slider, 'step', 0, max_steps - 1,
                           valinit=-1, valfmt=u'%d')

    def update(self, step):
        step = int(step)
        self.ax.lines = []
        self.ax.texts = []

        # Redefine the step and label from the self.sims list
        # so we can plot it using the self.plot_neb_curves
        for names in self.sims:
            names[self.step_index] = step
            names[self.label_index] = 'Step {}'.format(step)

        self.ax.legend_ = None

        self.plot_neb_curves()

        self.fig.canvas.draw_idle()


class plot_dist_vs_sknum(BaseNEBPlot):
    """

    simulation      ::       A simulation object from Finmag or Fidimag. We use
                             it to load the magnetisation fields from the NPYs
                             files

    sims   ::  An array of arrays with 4 elements each: The array has
               the structure

                   [[dms_file, label, step, simname, *energy_file], [%], ...]

               * The dms_file can be specified by the whole route to the
                 file, e.g. 'data/simulation/{simname}_dms.ndt'
               * It is assumed that the directory 'npys/{simname}_{step}'
                 exists. This is used to extract the magnetization
                 components from the images
               * The energy file is optional

    secondary_axis  ::      If the *energy_file was specified in any of the
                            sims lists, this option is necessary to show the
                            energy curve. Specify this argument as a two
                            element list, where the first argument is the
                            energy scale and the second one a label with the
                            energy units, e.g.
                                    [1.602e-19, 'eV']

    interpolate_energy  ::  This option can be added to the secondary_axis to
                            use an interpolation for the energy curve.
                            Specify this option as a two elements list with:
                            (i) the NEBM class and (ii) te resolution of the
                            interpolation.

    """
    # Inherit the doc from the BasePlot
    __doc__ += BaseNEBPlot.__doc__

    def __init__(self, simulation, *args, **kwargs):
        super(plot_dist_vs_sknum, self).__init__(*args,
                                                 **kwargs
                                                 )

        if BACKEND == 'FIDIMAG':
            # Methods are only defined for Fidimag simulations
            self.sk_number_method = kwargs.get('sk_number_method',
                                               'FiniteSpinChirality')

        self.ax2 = None

        # If the energy file was specified in any of the lists, initiate
        # the secondary axis ax2 using the same x-axis than ax
        for names in self.sims:
            if len(names) == 5 and str(names[-1]).endswith('energy.ndt'):
                self.init_secondary_axis()
                break

        # Parameters to interpolate the energy curve if necessary
        if self.interpolate_energy:
            self.nebm_class = self.interpolate_energy[0]
            self.interp_res = self.interpolate_energy[1]

        (self.energy_index,
         self.dms_index,
         self.label_index,
         self.step_index,
         self.sim_index) = (4, 0, 1, 2, 3)

        # This can be a Finmag or Fidimag simulation
        self.simulation = simulation

        # Secondary axis ------------------------------------------------------

        if self.ax2:
            self.scale = self.secondary_axis[0]

        # ---------------------------------------------------------------------

        # Plot the sk number curves and (if corresponds) the energy curves
        self.plot_neb_curves()

        # Decorate Plots ------------------------------------------------------

        # Top and bottom plots ------------------------------------------------
        self.insert_images_plots()

        # Decorations ---------------------------------------------------------

         # Limits
        self.set_limits_and_ticks()

        self.decorate_plot(DISTANCE_LABEL, r'$ Q $')

        if self.ax2:
            self.secondary_axis_label += (r'Energy ('
                                          + '{}'.format(self.secondary_axis[1])
                                          + r')')
            self.ax2.set_ylabel(self.secondary_axis_label)
            if self.ylim_en:
                self.ax2.set_ylim(self.ylim_en)

            # Necessary in case that an energy curve is interpolated, since
            # its legend only shows the markers
            for mark in self.ax.get_legend().legendHandles:
                if mark.get_linewidth() < 0.1:
                    mark.set_linewidth(2)
                    mark.lineStyles['-'] = '_draw_solid'

    def plot_neb_curves(self):

        for names in self.sims:

            x_data = self.generate_x_data_from_dms(names)
            npys_flist = self.generate_npys_file_list(names)

            # Now store all the skyrmion numbers computed with finmag /fidimag
            # for each image in a python array. The rows represent different
            # images
            y_data = []
            for files in npys_flist:
                # Load the magnetization profile
                self.simulation.set_m(np.load('{}npys/{}_{}/{}'.format(
                    self.rel_folder,
                    names[self.sim_index],
                    names[self.step_index],
                    files
                    )
                    ))

                if BACKEND == 'FINMAG':
                    y_data.append(-self.simulation.skyrmion_number())
                elif BACKEND == 'FIDIMAG':
                    y_data.append(self.simulation.skyrmion_number(method=self.sk_number_method))
                else:
                    raise ValueError('Choose between FINMAG or FIDIMAG')

            # Plot an energy curve in ax2 if the energy file is in the present
            # list (names), and redefine the legend
            if len(names) == 5 and str(names[-1]).endswith('energy.ndt'):
                # Make a white box to simulate a title in the legend
                # The idea is to make something like:
                #   Title 1
                # o Sk Number
                # v Energy
                #   Title 2
                # x Sk Number
                # ^ Energy
                ghost_box = matplotlib.patches.Rectangle((0, 0), 0, 0,
                                                         color='white')
                self.plots.append(ghost_box)
                self.labels.append(names[self.label_index])

                self.secondary_axis_energy_plot(x_data, names)

            # Plot the sk number curve
            y_data = np.array(y_data)
            self.plot_sk_number_curve(x_data, y_data, names)

    def plot_sk_number_curve(self, x_data, y_data, names):
        p, = self.ax.plot(x_data, y_data,
                          marker=next(self.markers),
                          lw=2,
                          # linestyle='',
                          # markerfacecolor=linec.get_color(),
                          markeredgecolor='black',
                          markeredgewidth=1.5,
                          # label=names[3]
                          )
        self.plots.append(p)

        if len(names) == 5 and str(names[-1]).endswith('energy.ndt'):
            self.labels.append(r'Sk Number')
        else:
            self.labels.append(names[self.label_index])

        # Annotate the numbers if they were specified
        if self.num_labels and self.sims.index(names) in self.num_labels:
            if isinstance(self.num_scale, list):
                num_index = self.num_labels.index(self.sims.index(names))
                scale = self.num_scale[num_index]
            else:
                scale = self.num_scale

            self.annotate_numbers(self.ax, x_data, y_data,
                                  scale=scale,
                                  ylim=self.ylim
                                  )


class plot_dist_vs_m(BaseNEBPlot):
    """

    simulation      ::       A simulation object from Finmag or Fidimag. We use
                             it to load the magnetisation fields from the NPYs
                             files

    m_components    ::       A tuple or list with 3 booleans indicating which
                             of the (m_x, m_y, m_z) components are going to be
                             plotted for every simulation specified in the
                             sims list

    sims  ::   An array of arrays with 4 elements each:
               The array has the structure

               [[dms_file, label, step, simname, *energy_file], [%], ...]

               * The dms_file can be specified by the whole route to the
                 file, e.g. 'data/simulation/{simname}_dms.ndt'
               * It is assumed that the directory 'npys/{simname}_{step}'
                 exists. This is used to extract the magnetization
                 components from the images
               * The energy file is optional

    secondary_axis  ::      If the *energy_file was specified in any of the
                            sims lists, this option is necessary to show the
                            energy curve. Specify this argument as a two
                            element list, where the first argument is the
                            energy scale and the second one a label with the
                            energy units, e.g.
                                    [1.602e-19, 'eV']

    interpolate_energy  ::  This option can be added to the secondary_axis to
                            use an interpolation for the energy curve.
                            Specify this option as a two elements list with:
                            (i) the NEBM class and (ii) te resolution of the
                            interpolation.

    """
    # Inherit the doc from the BasePlot
    __doc__ += BaseNEBPlot.__doc__

    def __init__(self, simulation, m_components, *args, **kwargs):

        super(plot_dist_vs_m, self).__init__(*args,
                                             **kwargs
                                             )

        for names in self.sims:
            if len(names) == 5 and str(names[-1]).endswith('energy.ndt'):
                self.init_secondary_axis()
                break

        if self.interpolate_energy:
            self.nebm_class = self.interpolate_energy[0]
            self.interp_res = self.interpolate_energy[1]

        self.m_components = m_components

        if not kwargs.get('num_scale'):
            kwargs['num_scale'] = [[0.5 if v is True else 0 for v in self.m_components]]

        self.m_labels = [r'$m_{x}$', r'$m_{y}$', r'$m_{z}$']

        (self.energy_index,
         self.dms_index,
         self.label_index,
         self.step_index,
         self.sim_index) = (4, 0, 1, 2, 3)

        # This can be a Finmag or Fidimag simulation
        self.simulation = simulation

        # Secondary axis ------------------------------------------------------
        if self.ax2:
            self.scale = self.secondary_axis[0]

        # ---------------------------------------------------------------------

        self.plot_neb_curves()

        # Decorate Plots ------------------------------------------------------

        # OPTIONAL Initial State Energy Plots
        # self.plot_initial_energy_curve()

        # Top and bottom plots ------------------------------------------------
        self.insert_images_plots()

        # Decorations ---------------------------------------------------------
        # Limits
        self.set_limits_and_ticks()

        self.decorate_plot(DISTANCE_LABEL, r'$ \langle m_{i} \rangle $')

        if self.ax2:
            self.secondary_axis_label += (r'Energy ('
                                          + '{}'.format(self.secondary_axis[1])
                                          + r')')
            self.ax2.set_ylabel(self.secondary_axis_label)
            if self.ylim_en:
                self.ax2.set_ylim(self.ylim_en)

            # Necessary in case that an energy curve is interpolated, since
            # its legend only shows the markers
            for mark in self.ax.get_legend().legendHandles:
                if mark.get_linewidth() < 0.1:
                    mark.set_linewidth(2)
                    mark.lineStyles['-'] = '_draw_solid'

    def plot_neb_curves(self):

        for names in self.sims:

            x_data = self.generate_x_data_from_dms(names)
            npys_flist = self.generate_npys_file_list(names)

            y_data = np.array([]).reshape(0, 3)
            for files in npys_flist:
                # Load the magnetization profile
                self.simulation.set_m(np.load('{}npys/{}_{}/{}'.format(
                    self.rel_folder,
                    names[self.sim_index],
                    names[self.step_index],
                    files
                    )
                    ))

                if BACKEND == 'FINMAG':
                    y_data = np.vstack((y_data,
                                        self.simulation.m_field.average()
                                        ))
                elif BACKEND == 'FIDIMAG':
                    y_data = np.vstack((y_data,
                                        self.simulation.compute_average()
                                        ))
                else:
                    raise ValueError('Choose between FINMAG or FIDIMAG')

            # Make a white box to simulate a title in the legend
            # The idea is to make something like:
            #   Title 1
            # o mx
            # v my
            #   Title 2
            # x mx
            # ^ my
            # ...
            # Or sk in the case of the skyrmion number
            ghost_box = matplotlib.patches.Rectangle((0, 0), 0, 0,
                                                     color='white')
            self.plots.append(ghost_box)
            self.labels.append(names[self.label_index])

            if len(names) == 5 and str(names[-1]).endswith('energy.ndt'):
                self.secondary_axis_energy_plot(x_data, names)

            # Number annotation of m curves -----------------------------------
            # The num_scale can be either a list of lists OR a list of floats.
            # num_scale must have the same number of elements than num_labels.
            # We first check if the curve index is in the num_labels list. If
            # so, we use its index to get the position in the num_scale list.
            #
            # If the num_scale[curve_index] is a list, we can either specify
            # a list of scales for every True component in self.m_components.
            # If it is just a float, we use the same scale for m_x, m_y, m_z
            #
            # Example: If we have two curves with
            #   self.m_components=(True, False, True)
            # (i.e. we only plot m_x and m_z) and:
            #   num_labels = [1]
            #   num_scale = [[1, -1]]
            #
            # Then, for the 1-th curve:
            #   num_scale_list = [1, 0, -1]
            #
            # Or if num_scale = [2], then
            #   num_scale_list = [2, 2, 2]
            # -------------------------------------------------
            # If num_labels = False, num_scale_list = [0, 0, 0]
            #
            # When num_scale_list[i] is zero, annotated numbers are not shown
            if self.num_labels and self.sims.index(names) in self.num_labels:
                num_index = self.num_labels.index(self.sims.index(names))
                if isinstance(self.num_scale[num_index], list):
                    num_iter = itertools.cycle(self.num_scale[num_index])
                    num_scale_list = [next(num_iter) if v is True else 0
                                      for v in self.m_components]
                else:
                    num_scale_list = [self.num_scale[num_index] if v is True
                                      else 0 for v in self.m_components]
            else:
                num_scale_list = [0] * 3
            # -----------------------------------------------------------------

            # Now plot the averages if they are specified in the arguments
            for i, m_label in enumerate(self.m_labels):
                if self.m_components[i]:
                    self.plot_m_average_curve(x_data, y_data, names, i, num_scale_list[i])

    def plot_m_average_curve(self, x_data, y_data, names, c_index, num_scale):
        p, = self.ax.plot(x_data, y_data[:, c_index],
                          marker=next(self.markers),
                          lw=2,
                          # linestyle='',
                          # markerfacecolor=linec.get_color(),
                          markeredgecolor='black',
                          markeredgewidth=1.5,
                          # label=names[3]
                          )
        self.plots.append(p)
        self.labels.append(self.m_labels[c_index])

        # See energy_curve plot function for an explanation
        self.annotate_numbers(self.ax, x_data, y_data[:, c_index],
                              scale=num_scale,
                              ylim=self.ylim,
                              )


# -----------------------------------------------------------------------------
# DISTANCES PLOT
# -----------------------------------------------------------------------------

class plot_distances(BaseNEBPlot):
    """

    Plot the distance between consecutive images, from i to i+1.  This is a 2D
    version.  The filenames needed in the inputs, are the files produced by the
    NEB simulation (''_energy.dt and ''_dms.ndt)

    Inputs:

    fnames     -->  An array of arrays with 3 elements each:
                    The array has the structure

                    ['distances_filename', 'label', step], [%], ...]

                    * Filenames can be specified with the whole route to
                      the file, e.g. 'data/simulations/neb_k1e10_energy.ndt'
                    * For the final state use step = -1

    """

    # Inherit the doc from the BasePlot
    __doc__ += BaseNEBPlot.__doc__

    def __init__(self, *args, **kwargs):

        setattr(self, 'secondary_axis', kwargs.get('secondary_axis', None))

        super(plot_distances, self).__init__(*args,
                                             **kwargs)

        (self.dms_index,
         self.label_index,
         self.step_index) = (0, 1, 2)

        self.plot_dms()

        # Top and bottom plots ------------------------------------------------
        self.insert_images_plots()

        if self.secondary_axis:
            self.generate_secondary_axis_energy_scale()

        # Decorations ---------------------------------------------------------
        self.decorate_plot(r'Image $i\rightarrow i+1$',
                           r'$|\bm{Y}_{i+1} - \bm{Y}_{i}|$',
                           )

    def plot_dms(self):
        # Main argument --------------------------------------------

        # Iterate for every element of the first argument which is an
        # array of arrays
        for names in self.sims:
            # Load the distances between consecutive images
            # from the dms file
            dms = np.loadtxt(names[0])[names[2]][1:]
            p, = self.ax.plot(dms,
                              marker=next(self.markers),
                              lw=2,
                              markeredgecolor='black',
                              markeredgewidth=1.5,
                              label=names[1]
                              )
            self.plots.append(p)
            self.labels.append(names[1])


# -----------------------------------------------------------------------------
# ENERGY BARRIER EVOLUTION ----------------------------------------------------
# -----------------------------------------------------------------------------

class plot_energy_barrier_evol(BaseNEBPlot):
    """
    Plot the energy barrier of the system as a function of the iteration steps
    The filenames needed in the inputs, are the files produced by the NEB
    simulation (''_energy.dt and ''_dms.ndt)

    Inputs:

    fnames     -->  An array of arrays with 3 elements each:
                    The array has the structure

                    [['energy_file1', 'energy_file2', ...], 'label', **image_range],
                        [%], ... ]

                    * img_range is OPTIONAL (see below for details)
                    * Filenames can be specified with the whole route to
                      the file, e.g. 'data/simulations/neb_k1e10_energy.ndt'
                    * Energy files must be in order, since this function
                      adds the iteration steps automatically from the total
                      number of steps from the previous file. E.g.

                      [energy_0-1000steps.ndt, energy_1000-2500steps.ndt, ...]

                ** image_range  A list or array with one OR two values representing
                                either: the initial and final images in the range where
                                the energy barrier is going to be computed
                                (diff between larger and smaller steps)
                                (images start at 0)

                                OR

                                a single value used as the reference image
                                for computing the Energy Barrier. e.g.
                                if [0] is used, the EB will be the difference
                                between the largest energy state and the energy
                                of the 0th image


                                By default, the function uses all the images
                                of the energy band

    """

    # Inherit the doc from the BasePlot
    __doc__ += BaseNEBPlot.__doc__

    def __init__(self, *args, **kwargs):

        setattr(self, 'secondary_axis', kwargs.get('secondary_axis', None))

        super(plot_energy_barrier_evol, self).__init__(*args,
                                                       **kwargs)

        (self.dms_index,
         self.label_index,
         self.step_index) = (0, 1, 2)

        # Redefine markers and default color cycle ----------------------------

        # lt = itertools.cycle(['-', '--', '-.', ':'])
        # [5, 3, 1] means, for example: a dash of length 5, a space of length 3
        #                               and  a dash of length 1 (like a point)
        self.dashes = itertools.cycle([(None, None), [5, 5], [12, 4, 8, 4],
                                      [5, 3, 1, 3], [8, 1, 8, 1, 2, 1, 2, 4],
                                      [1, 3]]
                                      )

        # Set the colour cycle according to the number of elements to be plotted
        # ax.set_color_cycle([cm(k) for k in np.linspace(0, 1, len(fnames) * 2)])
        d_palette_c = ['2C4C8F', 'FF5814', '000000', '542437', 'D91D2C',
                       '8F5536', '6F6F6F']
        d_palette_c = [npt_cm.hex_to_rgb(c) for c in d_palette_c]
        self.ax.set_prop_cycle(cycler('color', d_palette_c))

        # ---------------------------------------------------------------------

        self.plot_barrier_evolution()

        self.set_limits_and_ticks()

        # Top and bottom plots ------------------------------------------------
        self.insert_images_plots()

        if self.secondary_axis:
            self.generate_secondary_axis_energy_scale()

        # Decorations ---------------------------------------------------------
        self.decorate_plot(r'Iterations',
                           r'Energy  (' + self.scale_label + r')',
                           grid=False
                           )

        # Grid in light colour
        if self.grid:
            self.ax.grid(color='gray', linestyle='-', lw=0.3, alpha=0.3)

    def plot_barrier_evolution(self):

        # Iterate for every element of the first argument which is an
        # array of arrays
        for names in self.sims:
            # Load the energy data files from
            # from the _energy.ndt file
            energies = []
            num = 0

            for energy_file in names[0]:
                energies_tmp = np.loadtxt(energy_file)
                for line in energies_tmp:
                    # Compute the energy difference between the optional
                    # image_range in the last entry of the lists.

                    # If the list has two values (a range):
                    # Since the first values in the energy_file lines are step
                    # numbers, we add 1 to the first element in the list range.
                    # The last element has a 2 due to the step number AND also to
                    # consider the last image in image_range
                    #
                    # If the list has a single value:
                    # This will be used as the minimum
                    if len(names) == 3:
                        if len(names[2]) == 2:
                            en_barr = np.abs(np.max(line[1 + names[2][0]:
                                                         2 + names[2][1]]
                                                    )
                                             - np.min(line[1 + names[2][0]:
                                                           2 + names[2][1]]
                                                      )
                                             )

                        elif len(names[2]) == 1:
                            en_barr = np.abs(np.max(line[1:])
                                             - line[1 + names[2][0]]
                                             )
                    # Use all the images by default
                    elif len(names) == 2:
                        en_barr = np.abs(np.max(line[1:]) - np.min(line[1:]))
                    else:
                        print('ERROR: Lists must contain only 2 or 3 elements')

                    energies.append([num + line[0], en_barr])
                num += line[0]

            energies = np.array(energies)

            p, = self.ax.plot(energies[:, 0], energies[:, 1] / self.scale,
                              lw=2,
                              label=names[1])
            self.plots.append(p)
            self.labels.append(names[1])

        for l in self.ax.get_lines():
            l.set_dashes(next(self.dashes))


# -----------------------------------------------------------------------------
# ------------------------      UTILS       -----------------------------------
# -----------------------------------------------------------------------------

# Compute the average Energy Barrier over a range of iteration steps
def compute_energy_barrier(fnames, ebrange, image_range=None,
                           image_ref=None, scale=SCALE):
    """
    Compute the average Energy Barrier over a range of iteration steps
    The filenames needed in the inputs, are the files produced by the NEB
    simulation (''_energy.dt and ''_dms.ndt)

    Inputs:

    fnames      -->  A list of the filenames with the energy data of the
                     NEB function

                     ['energy_file1', 'energy_file2', ... ]

                     * Filenames can be specified with the whole route to
                       the file, e.g. 'data/simulations/neb_k1e10_energy.ndt'
                     * Energy files must be in order, since this function
                       adds the iteration steps automatically from the total
                       number of steps from the previous file. E.g.

                       [energy_0-1000steps.ndt, energy_1000-2500steps.ndt, ...]

    ebrange     -->  An array with two values: the initial step and the final
                     step of the range of energy values that are going
                     to be averaged

    image_range --> An OPTIONAL list or array with two values representing
                    the initial and final images in the range where
                    the energy barrier is going to be computed
                    (images start at 0).

                    By default, the function uses all the images
                    of the energy band

    image_ref  --> Specify an image number (starting from 0) to use as a
                   reference to compute the energy barrier

    """

    # Main argument --------------------------------------------

    energies = []
    # num = 0

    for energy_file in fnames:
        energies_tmp = np.loadtxt(energy_file)
        for line in energies_tmp[ebrange[0]:ebrange[1]]:
            if image_range:
                en_barr = np.abs(np.max(line[1 + image_range[0]:
                                             2 + image_range[1]]
                                        )
                                 - np.min(line[1 + image_range[0]:
                                               2 + image_range[1]]
                                          )
                                 )
            elif image_ref is not None:
                en_barr = np.abs(np.max(line[1:])
                                 - line[1 + image_ref]
                                 )
            else:
                en_barr = np.abs(np.max(line[1:]) - np.min(line[1:]))
            energies.append([en_barr])
        # num += line[0]

    energies = np.array(energies)
    return np.mean(energies) / scale
