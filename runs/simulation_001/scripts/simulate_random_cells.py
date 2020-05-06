import numpy as np
import os
from skimage import color
import matplotlib.pyplot as plt
import javabridge
import bioformats
import random
import argparse

javabridge.start_vm(class_path=bioformats.JARS)

##################################################################################
# Functions
##################################################################################


##################################################################################
# Main
##################################################################################
def main():
    parser = argparse.ArgumentParser('Digital filter using two color colocalization.')
    parser.add_argument('-nc', '--num_cells', dest = 'num_cells', type=int, default=100, help='Output filename containing plots')
    parser.add_argument('-nt', '--num_taxa', dest = 'num_taxa', type=int, default=2, help='Output filename containing plots')
    parser.add_argument('-cr', '--cell_radius', dest = 'cell_radius', type=int, default=50, help='Output filename containing plots')
    parser.add_argument('-cl', '--cell_length', dest = 'cell_length', type=int, default=100, help='Output filename containing plots')
    parser.add_argument('-cw', '--cell_width', dest = 'cell_width', type=int, default=20, help='Output filename containing plots')
    parser.add_argument('-sp', '--spacer', dest = 'spacer', type=int, default= 200, help='Output filename containing plots')

    args = parser.parse_args()

    # Initialize image
    row_size = int(np.round((args.num_cells)**(1/2)))
    dimension = int(row_size * (2*args.spacer + 1))
    image_dimensions = [dimension]*2
    image = np.zeros(image_dimensions)

    # Add cell centers with random IDs
    # Rods
    # r = int(np.round(args.cell_width/2))
    # l = args.cell_length - args.cell_width
    # kernel = np.zeros([args.cell_length]*2)
    # for ki in range(2*r+1):
    #     for kj in range(2*r+1):
    #         if ((ki - r)**2 + (kj - r)**2 <= r**2) or ():
    #             kernel[kj,ki] = 1.
    # Circles
    r = args.cell_radius
    kernel = np.zeros([2*r+1]*2)
    for ki in range(2*r+1):
        for kj in range(2*r+1):
            if (ki - r)**2 + (kj - r)**2 <= r**2:
                kernel[kj,ki] = 1.

    k = 1
    for i in range(1, row_size + 1):
        for j in range(1, row_size + 1):
            xi = (2*args.spacer + 1)*i - args.spacer
            yj = (2*args.spacer + 1)*j - args.spacer
            image[yj - r:yj + r + 1, xi - r:xi + r + 1] = kernel*k
            # cell_id = random.randint(1, args.num_taxa)
            # for xr in np.arange(xi - r, xi + r + 1):
            #     for yr in np.arange(yj - r, yj + r + 1):
            #         if (xi - xr)**2 + (yj - yr)**2 <= r**2:
            #             image[yr,xr] = k
            k += 1
    image_rgb = color.label2rgb(image, bg_label=0, bg_color=(0,0,0))
    plt.imshow(image_rgb)
    plt.show()
    plt.close()



if __name__ == '__main__':
    main()

javabridge.kill_vm()
