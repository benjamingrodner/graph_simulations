import os
import re
import argparse
import re
import glob
from math import pi

def remove_slash_from_dir(dir):
    # Make dir proper
    if dir[len(dir) - 1] == '/':
        dir = dir[:-1]
    else:
        dir = dir
    return(dir)

##################################################################################
# Main
##################################################################################
def main():
    parser = argparse.ArgumentParser('Digital filter using two color colocalization.')
    parser.add_argument('-of', '--output_folder', dest = 'output_folder', type=str, help='Output folder containing plots')
    parser.add_argument('-ns', '--num_simulations', dest = 'num_simulations', type=int, default=1, help='Output filename containing plots')
    args = parser.parse_args()

    # Parameters
    #
    # simulate_random_cells
    output_folder = remove_slash_from_dir(args.output_folder)
    num_cells = '200'
    num_orientation_bins = '8'
    spacer = '200'
    half_cell_length = '100'
    half_cell_width = '40'
    seg_random_names_list = ['{}/random_image_{}'.format(output_folder, i) for i in range(1, args.num_simulations + 1)]
    seg_names = " ".join(seg_names_list)
    seg_extension = '_seg.npy'
    seg_theta_extension = '_seg_theta.npy'
    theta_cmap = 'hsv'
    cell_props_extension = '_cell_props.csv'
    #
    # simulate_layered_orientation_cells
    seg_layered_orientation_names_list = ['{}/layered_orientation_image_{}'.format(output_folder, i) for i in range(1, args.num_simulations + 1)]
    seg_layered_orientation_names = " ".join(seg_layered_orientation_names_list)
    num_orientation_layers = '2'
    theta_difference = str(pi/4)
    #
    # build_graph_from_segmentation
    graph_extension = '_graph.npy'
    graph_on_seg_extension = '_graph_on_seg.png'
    radius = '250'
    #
    # plot_graph_on_seg
    graph_cmap = 'plasma'
    cell_cmap = 'Greens'
    edge_width = '5'
    vertex_size = '5'
    vertex_color = 'w'
    #
    # assign_random_taxa
    num_taxa = '2'
    random_taxa_sub_extension = '_random_taxa'
    cell_features_extension = '_cell_features.csv'
    #
    # assign_layered_taxa
    num_taxa_layers = '2'
    layered_taxa_sub_extension = '_layered_taxa'
    #
    # get_cell_feature_vector
    seg_random_orientation_random_taxa_names_list = [i + random_taxa_sub_extension for i in seg_random_names_list]
    seg_random_orientation_layered_taxa_names_list = [i + layered_taxa_sub_extension for i in seg_random_names_list]
    seg_layered_orientation_random_taxa_names_list = [i + random_taxa_sub_extension for i in seg_layered_orientation_names_list]
    seg_layered_orientation_layered_taxa_names_list = [i + layered_taxa_sub_extension for i in seg_layered_orientation_names_list]
    seg_names_combined = seg_random_orientation_random_taxa_names_list +\
                        seg_random_orientation_layered_taxa_names_list +\
                        seg_layered_orientation_random_taxa_names_list +\
                        seg_layered_orientation_layered_taxa_names_list
    #
    # message_passing
    cap_distance_weighting = half_cell_length
    cell_features_messaged_extension = '_cell_features_messaged.csv'
    #
    # umap_cell_feature_vector
    umap_taxa_extension = '_umap_taxa.png'
    umap_theta_extension = '_umap_theta.png'
    umap_yloc_extension = '_umap_yloc.png'
    taxa_cmap = 'spectral'
    yloc_cmap = 'bwr'

# Simulate cell segs
    # Random orientation and distribution
    save = 'T'
    shell_command = "python simulate_random_cells.py " +\
                    seg_random_names +\
                    " -sext " + seg_extension +\
                    " -sthext " + seg_theta_extension +\
                    " -thcm " + theta_cmap +\
                    " -cpext " + cell_props_extension +\
                    " -nc " + num_cells +\
                    " -nob " + num_orientation_bins +\
                    " -cl " + half_cell_length +\
                    " -cw " + half_cell_width +\
                    " -sp " + spacer +\
                    " -s " + save
    print('Generating random cells...\nExecuting shell command:\n{}'.format(shell_command))
    os.system(shell_command)

    # Simulate layers of orientation for cell segs with randomly distributed cells
    save = 'T'
    shell_command = "python simulate_layered_orientation_cells.py " +\
                    seg_layered_orientation_names +\
                    " -sext " + seg_extension +\
                    " -cpext " + cell_props_extension +\
                    " -nc " + num_cells +\
                    " -nol " + num_orientation_layers +\
                    " -td " + theta_difference +\
                    " -cl " + half_cell_length +\
                    " -cw " + half_cell_width +\
                    " -sp " + spacer +\
                    " -s " + save
    print('Generating random cells...\nExecuting shell command:\n{}'.format(shell_command))
    os.system(shell_command)

# Generate graphs
    save = 'T'
    shell_command = "python build_graph_from_segmentation.py " +\
                    seg_random_names +\
                    seg_layered_orientation_names +\
                    " -sext " + seg_extension +\
                    " -cpext " + cell_props_extension +\
                    " -gext " + graph_extension +\
                    " -gsext " + graph_on_seg_extension +\
                    " -r " + radius +\
                    " -s " + save
                    # " -gcm " + graph_cmap +\
                    # " -ccm " + cell_cmap +\
                    # " -ew " + edge_width +\
                    # " -vs " + vertex_size +\
                    # " -vc " + vertex_color
    print('Generating graphs...\nExecuting shell command:\n{}'.format(shell_command))
    os.system(shell_command)

# Plot graphs on cell seg
    save = 'T'
    shell_command = "python plot_graph_on_seg.py " +\
                    seg_random_names +\
                    seg_layered_orientation_names +\
                    " -gcm " + graph_cmap +\
                    " -ccm " + cell_cmap +\
                    " -ew " + edge_width +\
                    " -vs " + vertex_size +\
                    " -vc " + vertex_color +\
                    " -sext " + seg_extension +\
                    " -cpext " + cell_props_extension +\
                    " -gext " + graph_extension +\
                    " -gsext " + graph_on_seg_extension +\
                    " -s " + save
    print('Plotting graphs...\nExecuting shell command:\n{}'.format(shell_command))
    os.system(shell_command)


# Assign taxa to seg, save cell props and taxa seg
    # Random and layered orientation
    save = 'F'
    shell_command = "python assign_random_taxa.py " +\
                    seg_random_names +\
                    seg_layered_orientation_names +\
                    " -nt " + num_taxa +\
                    " -sext " + seg_extension +\
                    " -tsext " + random_taxa_sub_extension +\
                    " -cpext " + cell_props_extension +\
                    " -s " + save
    print('Generating graphs...\nExecuting shell command:\n{}'.format(shell_command))
    os.system(shell_command)

    # layered taxa, random orientation
    save = 'F'
    shell_command = "python assign_layered_taxa.py " +\
                    seg_random_names +\
                    seg_layered_orientation_names +\
                    " -nt " + num_taxa +\
                    " -ntl " + num_taxa_layers +\
                    " -sext " + seg_extension +\
                    " -tsext " + layered_taxa_sub_extension +\
                    " -cpext " + cell_props_extension +\
                    " -s " + save
    print('Generating graphs...\nExecuting shell command:\n{}'.format(shell_command))
    os.system(shell_command)


# Generate initial feature vector for cells
    # Random and layered orientation, random taxa
    save = 'F'
    shell_command = "python get_cell_feature_vector.py " +\
                    seg_names_combined +\
                    " -nt " + num_taxa +\
                    " -nob " + num_orientation_bins +\
                    " -sext " + seg_extension +\
                    " -cpext " + cell_props_extension +\
                    " -cfext " + cell_features_extension +\
                    " -s " + save
    print('Generating graphs...\nExecuting shell command:\n{}'.format(shell_command))
    os.system(shell_command)


# Use message passing to create feature vectors describing surrounding cells
    # Random and layered orientation, random taxa
    save = 'F'
    shell_command = "python message_passing.py " +\
                    seg_names_combined +\
                    " -nm " + num_messages +\
                    " -cdw " + cap_distance_weighting +\
                    " -cfext " + cell_features_extension +\
                    " -cfmext " + cell_features_messaged_extension +\
                    " -gext " + graph_extension +\
                    " -s " + save
    print('Generating graphs...\nExecuting shell command:\n{}'.format(shell_command))
    os.system(shell_command)

# Embed feature vector in two dimensional space using umap
    # Random and layered orientation, random taxa
    save = 'F'
    shell_command = "python umap_cell_feature_vector.py " +\
                    seg_names_combined +\
                    " -nm " + num_messages +\
                    " -cdw " + cap_distance_weighting +\
                    " -tsext " + random_taxa_sub_extension +\
                    " -cfext " + cell_features_extension +\
                    " -cfmext " + cell_features_messaged_extension +\
                    " -gext " + graph_extension +\
                    " -utxext " + umap_taxa_extension +\
                    " -uthext " + umap_theta_extension +\
                    " -uyext " + umap_yloc_extension +\
                    " -txcm " + taxa_cmap +\
                    " -thcm " + theta_cmap +\
                    " -ylcm " + yloc_cmap +\
                    " -s " + save
    print('Generating graphs...\nExecuting shell command:\n{}'.format(shell_command))
    os.system(shell_command)

    return

if __name__ == '__main__':
    main()
