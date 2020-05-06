import numpy as np
import pandas as pd
import argparse
from skimage.future import graph
from skimage import filters, color, io

##################################################################################
# Functions
##################################################################################



##################################################################################
# Main
##################################################################################
def main():
    parser = argparse.ArgumentParser('Digital filter using two color colocalization.')
    parser.add_argument('seg_names', type=str, nargs = '+', help='Output filename containing plots')
    # parser.add_argument('-ns', '--num_simulations', dest = 'num_simulations', type=int, default=1, help='Output filename containing plots')
    # parser.add_argument('-nc', '--num_cells', dest = 'num_cells', type=int, default=200, help='Output filename containing plots')
    # parser.add_argument('-cr', '--cell_radius', dest = 'cell_radius', type=int, default=50, help='Output filename containing plots')
    # parser.add_argument('-cl', '--half_cell_length', dest = 'half_cell_length', type=int, default=100, help='Output filename containing plots')
    # parser.add_argument('-cw', '--half_cell_width', dest = 'half_cell_width', type=int, default=40, help='Output filename containing plots')
    # parser.add_argument('-sp', '--spacer', dest = 'spacer', type=int, default= 200, help='Output filename containing plots')
    parser.add_argument('-s', '--save', dest = 'save', type=str, default= 'T', help='Output filename containing plots')
    parser.add_argument('-oext', '--output_extension', dest = 'output_extension', type=str, help='Output filename containing plots')
    parser.add_argument('-sext', '--seg_extension', dest = 'seg_extension', type=str, help='Output filename containing plots')
    args = parser.parse_args()

    for seg_name in args.seg_names:
        seg_filename = seg_name + args.seg_extension
        seg = np.load(seg_filename)
        edge_map = skimage.filters.sobel(segmentation > 0)
        rag = graph.rag_boundary(seg, edge_map)
        edges_rgb = color.gray2rgb(edge_map)
        lc = graph.show_rag(seg, rag, edges_rgb, img_cmap=None, edge_cmap='viridis', edge_width=1.2)
        io.show()
    return

if __name__ == '__main__':
    main()
