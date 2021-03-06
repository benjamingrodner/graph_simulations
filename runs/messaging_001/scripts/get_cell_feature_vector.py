import numpy as np
import pandas as pd
import argparse
# from skimage.future import graph
# from skimage import filters, color, io
import matplotlib.pyplot as plt
# from skimage.segmentation import find_boundaries
import random
from math import pi

##################################################################################
# Functions
##################################################################################

def plot_seg(seg, rgb, colors, save, output_filename):
    fig = plt.figure()
    fig.set_size_inches(20,20)
    ax = fig.add_subplot(111)
    if rgb == 'T':
        if colors == 'all':
            seg_rgb = color.label2rgb(seg, bg_label=0, bg_color=(0,0,0))
        else:
            seg_rgb = color.label2rgb(seg, colors = colors, bg_label=0, bg_color=(0,0,0))
        ax.imshow(seg_rgb)
    else:
        im = ax.imshow(seg, cmap=colors)
        pos = ax.get_position().bounds
        cax = fig.add_axes([pos[0] - 0.02, pos[1], 0.2, 0.02])
        cbar = fig.colorbar(im, cax=cax, ticks=[-pi, 0, pi], orientation='vertical')
        cbar.ax.set_xticklabels(['-pi', '0', 'pi'])
    if args.save == 'T':
        plt.savefig(output_png_filename, bbox_inches = 'tight', transparent = True)
    else:
        plt.show()
    plt.close()


##################################################################################
# Main
##################################################################################
def main():
    parser = argparse.ArgumentParser('Digital filter using two color colocalization.')
    parser.add_argument('seg_names', type=str, nargs = '+', help='Output filename containing plots')
    parser.add_argument('-nt', '--num_taxa', dest = 'num_taxa', type=int, default=2, help='Output filename containing plots')
    parser.add_argument('-nob', '--num_orientation_bins', dest = 'num_orientation_bins', type=int, default=2, help='Output filename containing plots')
    parser.add_argument('-td', '--theta_difference', dest = 'theta_difference', type=float, default=pi/4, help='Output filename containing plots')
    # parser.add_argument('-nc', '--num_cells', dest = 'num_cells', type=int, default=200, help='Output filename containing plots')
    # parser.add_argument('-cr', '--cell_radius', dest = 'cell_radius', type=int, default=50, help='Output filename containing plots')
    # parser.add_argument('-cl', '--half_cell_length', dest = 'half_cell_length', type=int, default=100, help='Output filename containing plots')
    # parser.add_argument('-cw', '--half_cell_width', dest = 'half_cell_width', type=int, default=40, help='Output filename containing plots')
    # parser.add_argument('-sp', '--spacer', dest = 'spacer', type=int, default= 200, help='Output filename containing plots')
    # parser.add_argument('-r', '--radius', dest = 'radius', type=int, default = 500, help='Output filename containing plots')
    parser.add_argument('-s', '--save', dest = 'save', type=str, default= 'T', help='Output filename containing plots')
    # parser.add_argument('-gext', '--graph_extension', dest = 'graph_extension', type=str, help='Output filename containing plots')
    # parser.add_argument('-gsext', '--graph_on_seg_extension', dest = 'graph_on_seg_extension', type=str, help='Output filename containing plots')
    # parser.add_argument('-sext', '--seg_extension', dest = 'seg_extension', type=str, help='Output filename containing plots')
    parser.add_argument('-tsext', '--taxa_sub_extension', dest = 'taxa_sub_extension', type=str, help='Output filename containing plots')
    # parser.add_argument('-sthext', '--seg_theta_extension', dest = 'seg_theta_extension', type=str, help='Output filename containing plots')
    # parser.add_argument('-thcm', '--theta_cmap', dest = 'theta_cmap', type=str, help='Output filename containing plots')
    parser.add_argument('-cpext', '--cell_props_extension', dest = 'cell_props_extension', type=str, help='Output filename containing plots')
    parser.add_argument('-cfext', '--cell_features_extension', dest = 'cell_features_extension', type=str, help='Output filename containing plots')
    args = parser.parse_args()

    # r = args.radius
    # theta_bins = np.linspace(-pi - 1e-5, pi + 1e-5,args.num_orientation_bins + 1)
    # Input theta values in range [-pi/2, pi/2]
    theta_bins = [-pi/2]
    for t in range(args.num_orientation_bins):
        theta_bins.append(theta_bins[t] + args.theta_difference)
    print('theta_bins\n', theta_bins)
    for seg_name in args.seg_names:
        # seg_name_taxa = seg_name + args.taxa_sub_extension
        # seg_filename = seg_name_taxa + args.seg_extension
        # seg = np.load(seg_filename).astype(int)
        cell_properties_filename = seg_name + args.cell_props_extension
        cell_properties = pd.read_csv(cell_properties_filename)
        # cell_props_taxa_column = []
        # map_dict_taxa = {}
        cell_features = np.zeros([cell_properties.shape[0], args.num_taxa + args.num_orientation_bins])
        for index, row in cell_properties.iterrows():
            id, taxon, theta = row[['id','taxon', 'theta']]
            id = int(id)
            taxon = int(taxon)
            # # asssign random taxa and write to feature vector and cell_properties and mapping dictionary
            # taxon = random.random(1,args.num_taxa)
            cell_features[id - 1, taxon - 1] = 1
            # cell_props_taxa_column.append(taxon)
            # map_dict_taxa[id] = taxon
            # Assign theta feature
            # print('theta, theta_bins',theta, theta_bins)
            theta_bin_num = int(np.digitize(theta, theta_bins))
            cell_features[id - 1, args.num_taxa - 1 + theta_bin_num] = 1
            # if index in list(range(3)):
            #     print('theta',theta)
            #     print('theta_bin_num',theta_bin_num)
            # map_dict_theta[id] = theta
        # # overwrite cell properties file
        # cell_properties['taxon'] = cell_props_taxa_column
        if args.save == 'T':
            # cell_properties_taxa_filename = seg_name + args.taxa_sub_extension + args.cell_props_extension
            # cell_properties.to_csv(cell_properties_taxa_filename)
            cell_features_filename = seg_name + args.cell_features_extension
            print(cell_features_filename)
            np.save(cell_features_filename, cell_features)
        else:
            print('cell_features.shape', cell_features.shape)
        # Write new taxa and orientation segmentations
        # seg_taxa = copy(seg)
        # for k, v in map_dict_taxa.items(): seg_taxa[seg==k] = v
        # seg_theta = copy(seg)
        # for k, v in map_dict_theta.items(): seg_theta[seg==k] = v
        # Plot segmentations
        # seg_taxa_filename = seg_name_taxa + args.seg_extension
        # plot_seg(seg_taxa, rgb='T' 'all', args.save, seg_taxa_filename)
        # cmc = plt.cm.get_cmap(args.theta_cmap)
        # clrs = cmc(np.arange(20,230))
        # seg_theta_filename = seg_name_taxa + args.seg_theta_extension
        # plot_seg(seg_theta, rgb='F', args.theta_cmap, args.save, seg_theta_filename)


    return

if __name__ == '__main__':
    main()
