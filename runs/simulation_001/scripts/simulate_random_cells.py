import numpy as np
import os
from skimage import color
import matplotlib.pyplot as plt
import javabridge
import bioformats
import random

javabridge.start_vm(class_path=bioformats.JARS)

##################################################################################
# Functions
##################################################################################


##################################################################################
# Main
##################################################################################
def main():
    parser = argparse.ArgumentParser('Digital filter using two color colocalization.')
    parser.add_argument('-nc', '--num_cells', dest = 'num_cells', type=int, default=10000, help='Output filename containing plots')
    parser.add_argument('-nt', '--num_taxa', dest = 'num_taxa', type=int, default=2, help='Output filename containing plots')
    parser.add_argument('-cr', '--cell_radius', dest = 'cell_radius', type=int, default=10, help='Output filename containing plots')
    parser.add_argument('-sp', '--spacer', dest = 'spacer', type=int, default=25, help='Output filename containing plots')

    args = parser.parse_args()

    # Initialize image
    row_size = np.round((args.num_cells)**(1/2))
    dimension = row_size * (2*args.spacer + 1)
    image_dimensions = [dimension]*2
    image = np.zeros(image_dimensions)

    # Add cell centers with random IDs
    r = args.cell_radius
    k = 1
    for i in range(row_size):
        for j in range(row_size):
            xi = (args.spacer + 1)*i
            yj = (args.spacer + 1)*j
            # cell_id = random.randint(1, args.num_taxa)
            for xr in np.arange(xi - r, xi + r + 1):
                for yr in np.arange(yj - r, yj + r + 1):
                    if (xi - xr)**2 + (yj - yr)**2 <= r**2:
                        image[yr,xr] = k
            k += 1
    image_rgb = color.label2rgb(image, bg_label=0, bg_color=(0,0,0))
    plt.imshow(image_rgb)
    plt.show()
    plt.close()



if __name__ == '__main__':
    main()

javabridge.kill_vm()
