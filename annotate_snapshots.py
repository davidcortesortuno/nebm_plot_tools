import sys

import os, shutil
import matplotlib.pyplot as plt
import matplotlib
import glob
import numpy as np
import re


def annotate_snapshots(pngs_folder_path, fontsize=26, color='black'):

    if not pngs_folder_path.endswith('/'):
        pngs_folder_path += '/'

    pngs_list_full = sorted(glob.glob(pngs_folder_path + '*.png'),
                            key=lambda f: int(re.search(r'\d+(?=\.)', f).group(0))
                            )
    pngs_list_rel = sorted(os.listdir(pngs_folder_path),
                           key=lambda f: int(re.search(r'\d+(?=\.)', f).group(0))
                           )
    # file_pattern = re.search(r'.*(?=_\d+\.png)', pngs_list_rel[0]).group(0)

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    fig.axes[0].get_xaxis().set_visible(False)
    fig.axes[0].get_yaxis().set_visible(False)

    plot = ax.imshow(matplotlib.image.imread(pngs_list_full[0]))
    text = ax.text(0.99, 0.98,
                   r'',
                   transform=ax.transAxes,
                   va='top', ha='right',
                   fontsize=fontsize,
                   color=color
                   )

    for i, _file in enumerate(pngs_list_full):

        image_num = int(re.search(r'(?<=_)\d+(?=\.png)', _file).group(0))

        image = matplotlib.image.imread(_file)

        plot.set_data(image)
        text.set_text('{}'.format(image_num)
                      )
        plt.draw()

        plt.savefig(_file,
                    bbox_inches='tight', pad_inches=0,
                    # dpi=200
                    )

        # OR
        # extent = ax.get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted())
        # plt.savefig(outputname, bbox_inches=extent)

    # To avoid showing the last snapshot in the notebook
    plt.close()
