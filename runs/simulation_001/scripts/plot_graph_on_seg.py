import numpy as np
import pandas as pd
import argparse
from skimage.future import graph
from skimage import filters, color, io
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries

##################################################################################
# Functions
##################################################################################

def plot_graph_on_seg(cell_cmap, seg, graph_cmap, adjacency_matrix, cell_properties, edge_width, vertex_color, vertex_size, graph_on_seg_extension, save):
    fig = plt.figure()
    fig.set_size_inches(20,20)
    ax = fig.add_subplot(111)
    # Show Seg
    cmc = plt.cm.get_cmap(cell_cmap)
    clrs = cmc(range(20,230))
    seg_color = color.label2rgb(seg, colors = clrs, bg_label=0, bg_color=(0,0,0))
    ax.imshow(seg_color)
    # ax.imshow(seg, cmap = 'Reds', clim = (0, adjacency_matrix.shape[0] + 20))
    # Get color mapping for distance values
    # print('args.graph_cmap',args.graph_cmap)
    cmg = plt.cm.get_cmap(graph_cmap)
    dist_max = np.max(adjacency_matrix)
    adj_mat_norm = adjacency_matrix / dist_max
    norm_min = np.min(adj_mat_norm)
    adj_mat_cmapped = -adj_mat_norm + 1 + norm_min
    # Plot Graph
    for row in range(adjacency_matrix.shape[0]):
        for col in range(row, adjacency_matrix.shape[0]):
            if adjacency_matrix[row,col] > 0:
                # id_c0 = str(int(row+1))
                x0 = cell_properties.loc[cell_properties.id == row + 1, 'x'].values[0]
                y0 = cell_properties.loc[cell_properties.id == row + 1, 'y'].values[0]
                x1 = cell_properties.loc[cell_properties.id == col + 1, 'x'].values[0]
                y1 = cell_properties.loc[cell_properties.id == col + 1, 'y'].values[0]
                x = [x0, x1]
                y = [y0, y1]
                cmap_value = adj_mat_cmapped[row, col]
                clr = cmg(cmap_value)
                # print('x,y,clr,args.edge_width)',x,y,clr,args.edge_width)
                ax.plot(x, y, c = clr, lw = edge_width)
                ax.plot(x0,y0, c = vertex_color, marker = '.', ms = vertex_size)
    if save == 'T':
        graph_on_seg_output_filename = seg_name + graph_on_seg_extension
        plt.savefig(graph_on_seg_output_filename, bbox_inches = 'tight', transparent = True)
    else:
        plt.show()
    plt.close()


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
    # parser.add_argument('-r', '--radius', dest = 'radius', type=int, default = 500, help='Output filename containing plots')
    parser.add_argument('-gcm', '--graph_cmap', dest = 'graph_cmap', type=str, default = 'inferno', help='Output filename containing plots')
    parser.add_argument('-ccm', '--cell_cmap', dest = 'cell_cmap', type=str, default = 'Blues', help='Output filename containing plots')
    parser.add_argument('-vs', '--vertex_size', dest = 'vertex_size', type=int, default = 2, help='Output filename containing plots')
    parser.add_argument('-vc', '--vertex_color', dest = 'vertex_color', type=str, default = 'w', help='Output filename containing plots')
    parser.add_argument('-ew', '--edge_width', dest = 'edge_width', type=int, default = 2, help='Output filename containing plots')
    parser.add_argument('-s', '--save', dest = 'save', type=str, default= 'T', help='Output filename containing plots')
    parser.add_argument('-gext', '--graph_extension', dest = 'graph_extension', type=str, help='Output filename containing plots')
    parser.add_argument('-gsext', '--graph_on_seg_extension', dest = 'graph_on_seg_extension', type=str, help='Output filename containing plots')
    parser.add_argument('-sext', '--seg_extension', dest = 'seg_extension', type=str, help='Output filename containing plots')
    parser.add_argument('-cpext', '--cell_props_extension', dest = 'cell_props_extension', type=str, help='Output filename containing plots')
    args = parser.parse_args()

    for seg_name in args.seg_names:
        # Load seg
        seg_filename = seg_name + args.seg_extension
        seg = np.load(seg_filename).astype(int)
        # Load vertex list
        cell_properties_filename = seg_name + args.cell_props_extension
        cell_properties = pd.read_csv(cell_properties_filename)
        # load adjacency matrix
        adjacency_matrix_filename = seg_name + args.graph_extension
        adjacency_matrix = np.load(adjacency_matrix_filename)
        plot_graph_on_seg(args.cell_cmap, seg, args.graph_cmap, adjacency_matrix, cell_properties, args.edge_width,
                            args.vertex_color, args.vertex_size, args.graph_on_seg_extension, args.save):

        # Show graph on image
        # fig = plt.figure()
        # fig.set_size_inches(20,20)
        # ax = fig.add_subplot(111)
        # # Show Seg
        # cmc = plt.cm.get_cmap(args.cell_cmap)
        # clrs = cmc(range(20,230))
        # seg_color = color.label2rgb(seg, colors = clrs, bg_label=0, bg_color=(0,0,0))
        # ax.imshow(seg_color)
        # # ax.imshow(seg, cmap = 'Reds', clim = (0, adjacency_matrix.shape[0] + 20))
        # # Get color mapping for distance values
        # # print('args.graph_cmap',args.graph_cmap)
        # cmg = plt.cm.get_cmap(args.graph_cmap)
        # dist_max = np.max(adjacency_matrix)
        # adj_mat_norm = adjacency_matrix / dist_max
        # norm_min = np.min(adj_mat_norm)
        # adj_mat_cmapped = -adj_mat_norm + 1 + norm_min
        # # Plot Graph
        # for row in range(adjacency_matrix.shape[0]):
        #     for col in range(row, adjacency_matrix.shape[0]):
        #         if adjacency_matrix[row,col] > 0:
        #             # id_c0 = str(int(row+1))
        #             x0 = cell_properties.loc[cell_properties.id == row + 1, 'x'].values[0]
        #             y0 = cell_properties.loc[cell_properties.id == row + 1, 'y'].values[0]
        #             x1 = cell_properties.loc[cell_properties.id == col + 1, 'x'].values[0]
        #             y1 = cell_properties.loc[cell_properties.id == col + 1, 'y'].values[0]
        #             x = [x0, x1]
        #             y = [y0, y1]
        #             cmap_value = adj_mat_cmapped[row, col]
        #             clr = cmg(cmap_value)
        #             # print('x,y,clr,args.edge_width)',x,y,clr,args.edge_width)
        #             ax.plot(x, y, c = clr, lw = args.edge_width)
        #             ax.plot(x0,y0, c = args.vertex_color, marker = '.', ms = args.vertex_size)
        # if args.save == 'T':
        #     graph_on_seg_output_filename = seg_name + args.graph_on_seg_extension
        #     plt.savefig(graph_on_seg_output_filename, bbox_inches = 'tight', transparent = True)
        # else:
        #     plt.show()
        # plt.close()
    return

if __name__ == '__main__':
    main()
