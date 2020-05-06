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
    args = parser.parse_args()

    # Simulate random cells
    num_cells = 200
    random_cells_extension = '_random_cells.npy'
    shell_command = "python simulate_random_cells.py " +\
                    " -of " + args.output_folder +\
                    " -oext " + random_cells_extension
    print('Generating random cells...\nExecuting shell command:\n{}'.format(shell_command))
    os.system(shell_command)
    return

if __name__ == '__main__':
    main()
