import os
import re
import argparse
import re
import glob


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
    output_folder = remove_slash_from_dir(args.output_folder)
    num_cells = '200'
    seg_names_list = ['{}/random_cells_image_{}'.format(output_folder, i) for i in range(1, args.num_simulations + 1)]
    seg_names = " ".join(seg_names_list)
    seg_extension = '_seg.npy'
    cell_props_extension = '_cell_props.csv'
    graph_extension = '_graph.npy'
    graph_on_seg_extension = '_graph_on_seg.png'
    radius = '250'
    graph_cmap = 'plasma'
    cell_cmap = 'Greens'
    edge_width = '5'
    vertex_size = '5'
    vertex_color = 'w'

    # Simulate random cells
    shell_command = "python simulate_random_cells.py " +\
                    seg_names +\
                    " -sext " + seg_extension +\
                    " -cpext " + cell_props_extension +\
                    " -nc " + num_cells
    print('Generating random cells...\nExecuting shell command:\n{}'.format(shell_command))
    os.system(shell_command)

    # Generate graphs
    save = 'T'
    shell_command = "python build_graph_from_segmentation.py " +\
                    seg_names +\
                    " -sext " + seg_extension +\
                    " -cpext " + cell_props_extension +\
                    " -gext " + graph_extension +\
                    " -gsext " + graph_on_seg_extension +\
                    " -r " + radius +\
                    " -s " + save +\
                    " -gcm " + graph_cmap +\
                    " -ccm " + cell_cmap +\
                    " -ew " + edge_width +\
                    " -vs " + vertex_size +\
                    " -vc " + vertex_color
    print('Generating graphs...\nExecuting shell command:\n{}'.format(shell_command))
    os.system(shell_command)

    return

if __name__ == '__main__':
    main()
