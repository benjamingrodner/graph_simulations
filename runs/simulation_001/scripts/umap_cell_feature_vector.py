import numpy as np
# import pandas as pd
import argparse
# from skimage.future import graph
# from skimage import filters, color, io
import matplotlib.pyplot as plt
import umap
# from skimage.segmentation import find_boundaries
# import random
# from math import pi
# from itertools import compress

##################################################################################
# Functions
##################################################################################

def plot_umap(embedding, colors, colormap, discrete_values, save, output_filename):
    plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, cmap=colormap, s=5)
    plt.gca().set_aspect('equal', 'datalim')
    if isinstance(discrete_values, str):
        plt.colorbar()
    else:
        plt.colorbar(boundaries=np.arange(discrete_values + 1)-0.5).set_ticks(np.arange(discrete_values))
    plt.xlabel('UMAP 1', fontsize=24)
    plt.ylabel('UMAP 1', fontsize=24)
    if save == 'T':
        plt.savefig(output_filename)
    else:
        plt.show()
    plt.close()
    return


##################################################################################
# Main
##################################################################################
def main():
    parser = argparse.ArgumentParser('Digital filter using two color colocalization.')
    parser.add_argument('seg_names', type=str, nargs = '+', help='Output filename containing plots')
    parser.add_argument('-nm', '--num_messages', dest = 'num_messages', type=int, default=2, help='Output filename containing plots')
    parser.add_argument('-cdw', '--cap_distance_weighting', dest = 'cdw', type=str, default=40, help='Output filename containing plots')
    parser.add_argument('-nt', '--num_taxa', dest = 'num_taxa', type=int, default=2, help='Output filename containing plots')
    parser.add_argument('-nob', '--num_orientation_bins', dest = 'num_orientation_bins', type=int, default=2, help='Output filename containing plots')
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
    # parser.add_argument('-stext', '--seg_taxa_extension', dest = 'seg_taxa_extension', type=str, help='Output filename containing plots')
    # parser.add_argument('-sthext', '--seg_theta_extension', dest = 'seg_theta_extension', type=str, help='Output filename containing plots')
    # parser.add_argument('-thcm', '--theta_cmap', dest = 'theta_cmap', type=str, help='Output filename containing plots')
    parser.add_argument('-cpext', '--cell_props_extension', dest = 'cell_props_extension', type=str, help='Output filename containing plots')
    parser.add_argument('-tsext', '--taxa_sub_extension', dest = 'taxa_sub_extension', default = '', type=str, help='Output filename containing plots')
    parser.add_argument('-cfext', '--cell_features_extension', dest = 'cell_features_extension', type=str, help='Output filename containing plots')
    parser.add_argument('-cfmext', '--cell_features_messaged_extension', dest = 'cell_features_messaged_extension', type=str, help='Output filename containing plots')
    parser.add_argument('-gext', '--graph_extension', dest = 'graph_extension', type=str, help='Output filename containing plots')
    parser.add_argument('-utxext', '--umap_taxa_extension', dest = 'umap_taxa_extension', type=str, help='Output filename containing plots')
    parser.add_argument('-uthext', '--umap_theta_extension', dest = 'umap_theta_extension', type=str, help='Output filename containing plots')
    parser.add_argument('-uyext', '--umap_yloc_extension', dest = 'umap_yloc_extension', type=str, help='Output filename containing plots')
    parser.add_argument('-txcm', '--taxa_cmap', dest = 'taxa_cmap', type=str, help='Output filename containing plots')
    parser.add_argument('-thcm', '--theta_cmap', dest = 'theta_cmap', type=str, help='Output filename containing plots')
    parser.add_argument('-ylcm', '--yloc_cmap', dest = 'yloc_cmap', type=str, help='Output filename containing plots')
    args = parser.parse_args()

    for seg_name in args.seg_names:
        seg_name_taxa = seg_name + args.taxa_sub_extension
        cell_features_messaged_filename = seg_name_taxa + args.cell_features_messaged_extension
        cell_features_messaged = np.load(cell_features_messaged_filename)
        cell_properties_filename = seg_name_taxa + args.cell_props_extension
        cell_properties = pd.read_csv(cell_properties_filename)
        # adjacency_matrix_filename = seg_name_taxa + args.graph_extension
        # adjacency_matrix = np.load(adjacency_matrix_filename)
        # Weight distances such that shorter distances are more important
        # all weights in (0,1]
        reducer = umap.UMAP()
        reducer.fit(cell_features_messaged)
        embedding = reducer.transform(cell_features_messaged)

        # Plot Umaps
        taxa = cell_properties.loc[:,'taxon'].values
        ylocations = cell_properties.loc[:,'y'].values
        thetas = cell_properties.loc[:,'theta'].values
        #
        taxa_umap_filename = seg_name_taxa + args.umap_taxa_extension
        plot_umap(embedding, taxa, args.taxa_cmap, args.num_taxa, args.save, taxa_umap_filename)
        theta_umap_filename = seg_name_taxa + args.umap_theta_extension
        plot_umap(embedding, theta, args.theta_cmap, args.num_orientation_bins, args.save, theta_umap_filename)
        yloc_umap_filename = seg_name_taxa + args.umap_yloc_extension
        plot_umap(embedding, ylocations, args.yloc_cmap, 'F', args.save, yloc_umap_filename)
    return

if __name__ == '__main__':
    main()
