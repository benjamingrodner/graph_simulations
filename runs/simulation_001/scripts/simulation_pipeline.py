import os
import re
import argparse
import re
import glob

##################################################################################
# Main
##################################################################################
def main():
    parser = argparse.ArgumentParser('Digital filter using two color colocalization.')
    parser.add_argument('-of', '--output_folder', dest = 'output_folder', type=str, help='Output folder containing plots')
    parser.add_argument('-ns', '--num_simulations', dest = 'num_simulations', type=int, default=1, help='Output filename containing plots')
    args = parser.parse_args()

    # Parameters
    num_cells = '200'
    seg_names_list = ['{}/random_cells_image_{}'.format(args.output_folder, i) for i in range(1, args.num_simulations + 1)]
    seg_names = " ".join(seg_names_list)
    seg_extension = '_seg.npy'
    graph_extension = '_graph.npy'

    # Simulate random cells
    shell_command = "python simulate_random_cells.py " +\
                    seg_names +\
                    " -oext " + seg_extension +\
                    " -nc " + num_cells
    print('Generating random cells...\nExecuting shell command:\n{}'.format(shell_command))
    os.system(shell_command)

    # Generate graphs
    shell_command = "python build_graph_from_segmentation.py " +\
                    seg_names +\
                    " -sext " + seg_extension +\
                    " -oext " + graph_extension
    print('Generating graphs...\nExecuting shell command:\n{}'.format(shell_command))
    os.system(shell_command)

    return

if __name__ == '__main__':
    main()
