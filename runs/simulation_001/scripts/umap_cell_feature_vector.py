import numpy as np
import pandas as pd
import argparse
# from skimage.future import graph
# from skimage import filters, color, io
import matplotlib.pyplot as plt
import umap
# from skimage.segmentation import find_boundaries
# import random
from math import pi
# from itertools import compress

##################################################################################
# Functions
##################################################################################

# def plot_umap(embedding, colors, colormap, clim, discrete_values, save, output_filename):
#     if isinstance(clim, str):
#         plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, cmap=colormap, s=5)
#     else:
#         plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, clim=(clim[0],clim[1]), cmap=colormap, s=5)
#     plt.gca().set_aspect('equal', 'datalim')
#     if isinstance(discrete_values, str):
#         plt.colorbar()
#     else:
#         plt.colorbar(boundaries=np.arange(1, discrete_values + 2) - 0.5).set_ticks(colors)
#     plt.xlabel('UMAP 1', fontsize=24)
#     plt.ylabel('UMAP 2', fontsize=24)
#     if save == 'T':
#         plt.savefig(output_filename)
#     else:
#         plt.show()
#     plt.close()
#     return

def plot_umap(embedding, colors, colormap, type, num_discrete_values, save, output_png_filename):
    fig = plt.figure()
    fig.set_size_inches(10,10)
    ax = fig.add_subplot(111)
    s = 30
    if type == 'discrete':
        im = ax.scatter(embedding[:, 0], embedding[:, 1], c=colors, cmap=colormap, s=s)
        plt.gca().set_aspect('equal', 'datalim')
        pos = ax.get_position().bounds
        cax = fig.add_axes([pos[0] + pos[2] + 0.02, pos[1], 0.02, pos[3]])
        cbar = fig.colorbar(im, cax=cax, ticks=colors, boundaries = np.arange(1, num_discrete_values + 2) - 0.5, orientation='vertical')
    elif type == 'theta':
        im = ax.scatter(embedding[:, 0], embedding[:, 1], c=colors, vmin=-pi/2, vmax= pi/2, cmap=colormap, s=s)
        # im = ax.scatter(embedding[:, 0], embedding[:, 1], c=colors, cmap=colormap, clim = (-pi/2,pi/2), s=s)
        # im = ax.imshow(seg, cmap=colors, clim = (-pi/2,pi/2))
        plt.gca().set_aspect('equal', 'datalim')
        pos = ax.get_position().bounds
        cax = fig.add_axes([pos[0] + pos[2] + 0.02, pos[1], 0.02, pos[3]])
        # cbar = fig.colorbar(im, cax=cax, orientation='vertical')
        cbar = fig.colorbar(im, cax=cax, ticks=[-pi/2, 0, pi/2], orientation='vertical')
        cbar.ax.set_yticklabels(['-pi/2', '0', 'pi/2'])
    elif type == 'continuous':
        im = ax.scatter(embedding[:, 0], embedding[:, 1], c=colors, cmap=colormap, s=s)
        # im = ax.imshow(seg, cmap=colors, clim = (-pi/2,pi/2))
        plt.gca().set_aspect('equal', 'datalim')
        pos = ax.get_position().bounds
        cax = fig.add_axes([pos[0] + pos[2] + 0.02, pos[1], 0.02, pos[3]])
        cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_xlabel('UMAP 1', fontsize=24)
    ax.set_ylabel('UMAP 2', fontsize=24)
    ax.set_xticks([])
    ax.set_yticks([])
    if save == 'T':
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
    parser.add_argument('-nm', '--num_messages', dest = 'num_messages', type=int, default=2, help='Output filename containing plots')
    parser.add_argument('-cdw', '--cap_distance_weighting', dest = 'cdw', type=str, default=40, help='Output filename containing plots')
    parser.add_argument('-nt', '--num_taxa', dest = 'num_taxa', type=int, default=2, help='Output filename containing plots')
    parser.add_argument('-nob', '--num_orientation_bins', dest = 'num_orientation_bins', type=int, default=8, help='Output filename containing plots')
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
        # seg_name_taxa = seg_name + args.taxa_sub_extension
        cell_features_messaged_filename = seg_name + args.cell_features_messaged_extension
        cell_features_messaged = np.load(cell_features_messaged_filename)
        cell_properties_filename = seg_name + args.cell_props_extension
        cell_properties = pd.read_csv(cell_properties_filename)
        # adjacency_matrix_filename = seg_name + args.graph_extension
        # adjacency_matrix = np.load(adjacency_matrix_filename)
        # Weight distances such that shorter distances are more important
        # all weights in (0,1]
        reducer = umap.UMAP(n_neighbors = 20, min_dist = 0.5)
        reducer.fit(cell_features_messaged)
        embedding = reducer.transform(cell_features_messaged)

        # Plot Umaps
        taxa = cell_properties.loc[:,'taxon'].values
        ylocations = cell_properties.loc[:,'y'].values
        thetas = cell_properties.loc[:,'theta'].values
        # print('thetas',thetas[0:10])
        # print('embedding', embedding[0:10,:])
        #
        theta_umap_filename = seg_name + args.umap_theta_extension
        # theta_cmap = plt.cm.get_cmap(args.theta_cmap)
        theta_clim = [-pi/2,pi/2]
        # print('theta_clim',theta_clim)
        plot_umap(embedding, thetas, args.theta_cmap, 'theta','F', args.save, theta_umap_filename)
        #
        taxa_umap_filename = seg_name + args.umap_taxa_extension
        taxa_cmap = plt.cm.get_cmap(args.taxa_cmap, args.num_taxa)
        plot_umap(embedding, taxa, taxa_cmap, 'discrete', args.num_taxa, args.save, taxa_umap_filename)
        #
        yloc_umap_filename = seg_name + args.umap_yloc_extension
        plot_umap(embedding, ylocations, args.yloc_cmap, 'continuous', 'F', args.save, yloc_umap_filename)
    return

if __name__ == '__main__':
    main()
