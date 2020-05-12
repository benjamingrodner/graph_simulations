import numpy as np
# import pandas as pd
import argparse
# from skimage.future import graph
# from skimage import filters, color, io
# import matplotlib.pyplot as plt
# from skimage.segmentation import find_boundaries
# import random
# from math import pi
# from itertools import compress

##################################################################################
# Functions
##################################################################################



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
    # parser.add_argument('-cpext', '--cell_props_extension', dest = 'cell_props_extension', type=str, help='Output filename containing plots')
    parser.add_argument('-tsext', '--taxa_sub_extension', dest = 'taxa_sub_extension', default = '', type=str, help='Output filename containing plots')
    parser.add_argument('-cfext', '--cell_features_extension', dest = 'cell_features_extension', type=str, help='Output filename containing plots')
    parser.add_argument('-cfmext', '--cell_features_messaged_extension', dest = 'cell_features_messaged_extension', type=str, help='Output filename containing plots')
    parser.add_argument('-gext', '--graph_extension', dest = 'graph_extension', type=str, help='Output filename containing plots')
    args = parser.parse_args()

    for seg_name in args.seg_names:
        # seg_name_taxa = seg_name + args.taxa_sub_extension
        cell_features_filename = seg_name + args.cell_features_extension
        cell_features = np.load(cell_features_filename)
        cell_features_messaged = copy(cell_features)
        adjacency_matrix_filename = seg_name + args.graph_extension
        adjacency_matrix = np.load(adjacency_matrix_filename)
        # Weight distances such that shorter distances are more important
        # all weights in (0,1]
        if not args.cap_distance_weighting == 'none':
            # Weight edges equally when they are closer than a certain distance
            cap_distance_weighting = args.cap_distance_weighting
            adjacency_matrix_remapped = copy(adjacency_matrix)
            adjacency_matrix_remapped[adjacency_matrix_remapped > cap_distance_weighting] == cap_distance_weighting
            adjacency_matrix_remapped = cap_distance_weighting * adjacency_matrix_remapped**-1
        else:
            dist_min = np.min(adjacency_matrix)
            adjacency_matrix_remapped = dist_min * adjacency_matrix**-1
        # How many message iterations
        message_array_previous = copy(cell_features)
        for m in range(args.num_messages):
            message_array_new = np.zeros(cell_features.shape)
            for cell in range(cell_features.shape[0]):
                edges = adjacency_matrix_remapped[cell,:]
                edges_bool = edges > 0
                # Get an array of distance weighted values for the edges
                edges_multiplier = edges[edges_bool][np.newaxis].T
                # Get the cell indices for the edges
                edges_id_list = np.where(edges_bool)[0]
                # For the first message, pass the theta values of the edge cells relative to the query cell
                if m == 0:
                    # separate the taxa feature values from the theta feature values
                    edges_message_taxa = message_array_previous[edges_id_list, 0:args.num_taxa]
                    edges_message_theta = message_array_previous[edges_id_list, args.num_taxa:]
                    # Get the edges' orientation features relative to the query cell
                    theta_cell = message_array_previous[cell, args.num_taxa:]
                    theta_shift = np.where(theta_cell > 0)[0][0]
                    edges_message_theta_shift = np.roll(edges_message_theta, -theta_shift, axis = 1)
                    # Recombine the feature values
                    edges_message = np.concatenate((edges_message_taxa, edges_message_theta_shift), axis = 1)
                else:
                    edges_message = message_array_previous[edges_id_list, :]
                # Weight the edges by distance
                edges_message_weighted = edges_message * edges_multiplier
                # Create a new message feature vector for the cell
                edges_message_compressed = np.sum(edges_message_weighted, axis = 0) / edges_message_weighted.shape[0]
                message_array_new[cell,:] = edges_message_compressed
            # Tack on the message info to the cell features
            cell_features_messaged = np.concatenate((cell_features_messaged, message_array_new), axis = 1)
            # New messages become the feature vectors referenced
            message_array_previous = message_array_new
        if args.save == 'T':
            cell_features_messaged_filename = seg_name + args.cell_features_messaged_extension
            np.save(cell_features_messaged_filename, cell_features_messaged)
        else:
            print("cell_features_messaged.shape",cell_features_messaged.shape)
    return

if __name__ == '__main__':
    main()
