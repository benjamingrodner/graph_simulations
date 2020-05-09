import numpy as np
import os
from skimage import color
import matplotlib.pyplot as plt
# import javabridge
# import bioformats
import random
import argparse
from math import pi
import re
import pandas as pd

# javabridge.start_vm(class_path=bioformats.JARS)

##################################################################################
# Functions
##################################################################################

# Get a list of kernels describing rod shaped kernels at various angles
def get_rod_kernels(half_cell_length, half_cell_width, theta_list):
    xc = half_cell_length
    yc = half_cell_length
    dw = half_cell_width
    dl = half_cell_length - half_cell_width
    kernel_list = []
    for theta in theta_list:
        # Define coordinates specifying points at the center of each edge
        if theta == pi/2 or theta == -pi/2:
            xpw0 = xc + dw
            ypw0 = yc
            xpw1 = xc - dw
            ypw1 = yc
            xpl0 = xc
            ypl0 = yc + dl
            xpl1 = xc
            ypl1 = yc - dl
        elif theta == 0:
            xpl0 = xc + dl
            ypl0 = yc
            xpl1 = xc - dl
            ypl1 = yc
            xpw0 = xc
            ypw0 = yc + dw
            xpw1 = xc
            ypw1 = yc - dw
        else:
            ml = np.tan(theta)
            mw = -1/ml
            # Length points
            xdl = np.sqrt(dl**2/(1+ml**2))
            ydl = xdl * ml
            xpl0 = xc + xdl
            ypl0 = yc + ydl
            xpl1 = xc - xdl
            ypl1 = yc - ydl
            # Width points
            xdw = np.sqrt(dw**2/(1+mw**2))
            ydw = xdw * mw
            xpw0 = xc + xdw
            ypw0 = yc + ydw
            xpw1 = xc - xdw
            ypw1 = yc - ydw
        # Write kernel
        kernel_dimension = half_cell_length*2 + 1
        kernel = np.zeros([kernel_dimension]*2).astype(int)
        for x in range(kernel_dimension):
            for y in range(kernel_dimension):
                # Bools
                if theta == pi/2 or theta == -pi/2:
                    bool_w0 = x - xpw0 <= 0
                    bool_w1 = x - xpw1 >= 0
                    bool_l0 = y - ypl0 <= 0
                    bool_l1 = y - ypl1 >= 0
                elif theta == 0:
                    bool_l0 = x - xpl0 <= 0
                    bool_l1 = x - xpl1 >= 0
                    bool_w0 = y - ypw0 <= 0
                    bool_w1 = y - ypw1 >= 0
                elif theta > 0:
                    bool_l0 = y - ypl0 - mw*(x - xpl0) <= 0
                    bool_l1 = y - ypl1 - mw*(x - xpl1) >= 0
                    bool_w0 = y - ypw0 - ml*(x - xpw0) >= 0
                    bool_w1 = y - ypw1 - ml*(x - xpw1) <= 0
                else:
                    bool_l0 = y - ypl0 - mw*(x - xpl0) >= 0
                    bool_l1 = y - ypl1 - mw*(x - xpl1) <= 0
                    bool_w0 = y - ypw0 - ml*(x - xpw0) <= 0
                    bool_w1 = y - ypw1 - ml*(x - xpw1) >= 0
                bool_c0 = (y - ypl0)**2 + (x - xpl0)**2 <= dw**2
                bool_c1 = (y - ypl1)**2 + (x - xpl1)**2 <= dw**2
                # Write kernel
                if bool_l0 and bool_l1 and bool_w0 and bool_w1 or bool_c0 or bool_c1:
                    kernel[y,x] = 1
        kernel_list.append(kernel)
    return(kernel_list)




##################################################################################
# Main
##################################################################################
def main():
    parser = argparse.ArgumentParser('Digital filter using two color colocalization.')
    # parser.add_argument('-ns', '--num_simulations', dest = 'num_simulations', type=int, default=1, help='Output filename containing plots')
    parser.add_argument('seg_names', type=str, nargs = '+', help='Output filename containing plots')
    parser.add_argument('-nc', '--num_cells', dest = 'num_cells', type=int, default=200, help='Output filename containing plots')
    # parser.add_argument('-cr', '--cell_radius', dest = 'cell_radius', type=int, default=50, help='Output filename containing plots')
    parser.add_argument('-cl', '--half_cell_length', dest = 'half_cell_length', type=int, default=100, help='Output filename containing plots')
    parser.add_argument('-cw', '--half_cell_width', dest = 'half_cell_width', type=int, default=40, help='Output filename containing plots')
    parser.add_argument('-sp', '--spacer', dest = 'spacer', type=int, default= 200, help='Output filename containing plots')
    parser.add_argument('-s', '--save', dest = 'save', type=str, default= 'T', help='Output filename containing plots')
    parser.add_argument('-of', '--output_folder', dest = 'output_folder', type=str, default= 'T', help='Output filename containing plots')
    parser.add_argument('-sext', '--seg_extension', dest = 'seg_extension', type=str, help='Output filename containing plots')
    parser.add_argument('-cpext', '--cell_props_extension', dest = 'cell_props_extension', type=str, help='Output filename containing plots')

    args = parser.parse_args()

    for seg_name in args.seg_names:
        # Initialize image
        row_size = int(np.round((args.num_cells)**(1/2)))
        dimension = int(row_size * (args.spacer + 1) + args.spacer)
        # dimension = int(row_size * (2*args.spacer + 1))
        image_dimensions = [dimension]*2
        image = np.zeros(image_dimensions)

        # Rods
        # Input theta values in range [-pi/2, pi/2]
        theta_list = np.arange(-pi/2, pi/2, pi/8)
        kernel_list = get_rod_kernels(args.half_cell_length, args.half_cell_width, theta_list)
        # # Circles
        # r = args.cell_radius
        # kernel = np.zeros([2*r+1]*2)
        # for ki in range(2*r+1):
        #     for kj in range(2*r+1):
        #         if (ki - r)**2 + (kj - r)**2 <= r**2:
        #             kernel[kj,ki] = 1.

        # Add randomly spaced cells with IDs and random orientations
        r = args.half_cell_length
        k = 1
        print('image.shape',image.shape)
        # Randomly placed cells
        cell_properties = pd.DataFrame()
        for c in range(args.num_cells):
            xi = random.randint(r ,dimension - r - 1)
            yj = random.randint(r, dimension - r - 1)
            krnd = random.randint(0, len(theta_list)-1)
            kernel = kernel_list[krnd]
            try:
                image[yj - r:yj + r + 1, xi - r:xi + r + 1] = image[yj - r:yj + r + 1, xi - r:xi + r + 1]*(kernel == 0) + kernel*k
            except:
                print('k', k)
                print('yj, xi', yj, xi)
            cell_properties = cell_properties.append({'id':k, 'theta':theta_list[krnd], 'x':xi, 'y':yj}, ignore_index = True)
            k +=1


        # # Evenly spaced cells with random jitter
        # for i in range(1, row_size + 1):
        #     for j in range(1, row_size + 1):
        #         xi = (args.spacer )*i
        #         # xi = (2*args.spacer + 1)*i - args.spacer
        #         yj = (args.spacer)*j
        #         # yj = (2*args.spacer + 1)*j - args.spacer
        #         krnd = random.randint(0, len(theta_list)-1)
        #         kernel = kernel_list[krnd]
        #         # print('k',k)
        #         # print('yjr - r, yjr + r + 1, xir - r, xir + r + 1',yjr - r, yjr + r + 1, xir - r, xir + r + 1)
        #         # loop = 'T'
        #         shift_length = args.spacer - args.half_cell_length
        #         # while loop == 'T':
        #         xrnd = random.randint(-shift_length,shift_length)
        #         yrnd = random.randint(-shift_length,shift_length)
        #         xir = xi + xrnd
        #         yjr = yj + yrnd
        #         try:
        #             image[yjr - r:yjr + r + 1, xir - r:xir + r + 1] = image[yjr - r:yjr + r + 1, xir - r:xir + r + 1]*(kernel == 0) + kernel*k
        #         except:
        #             print('k', k)
        #             print('j, i', j, i)
        #             print('yjr, xir', yjr, xir)
        #             # try:
        #             #     image[yjr - r:yjr + r + 1, xir - r:xir + r + 1] = image[yjr - r:yjr + r + 1, xir - r:xir + r + 1]*(kernel == 0) + kernel*k
        #             #     loop == 'F'
        #             # except:
        #             #     print('redo shift')
        #         # cell_id = random.randint(1, args.num_taxa)
        #         # for xr in np.arange(xi - r, xi + r + 1):
        #         #     for yr in np.arange(yj - r, yj + r + 1):
        #         #         if (xi - xr)**2 + (yj - yr)**2 <= r**2:
        #         #             image[yr,xr] = k
        #         k += 1
        # Show graph on image
        fig = plt.figure()
        fig.set_size_inches(20,20)
        ax = fig.add_subplot(111)
        image_rgb = color.label2rgb(image, bg_label=0, bg_color=(0,0,0))
        ax.imshow(image_rgb)
        if args.save == 'T':
            output_numpy_filename = seg_name + args.seg_extension
            np.save(output_numpy_filename, image)
            png_extension = re.sub('.npy','.png',args.seg_extension)
            output_png_filename = seg_name + png_extension
            plt.savefig(output_png_filename, bbox_inches = 'tight', transparent = True)
            cell_properties_filename = seg_name + args.cell_props_extension
            cell_properties.to_csv(cell_properties_filename)
        else:
            plt.show()
        plt.close()


if __name__ == '__main__':
    main()

# javabridge.kill_vm()
