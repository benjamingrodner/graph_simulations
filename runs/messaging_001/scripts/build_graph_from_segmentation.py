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

def build_adjacency_matrix(cell_properties, r, min_edges):
    # Create an adjacency matrix with weighted edges
    increase = int(r/4)
    adjacency_matrix = np.zeros([cell_properties.shape[0]]*2)
    for index, row in cell_properties.iterrows():
        id, xc, yc = row[['id','x','y']]
        id = int(id)
        adj_row = adjacency_matrix[index,:]
        if  adj_row[id-1] == 0:
            # Make sure each cell is conneted to the graph
            widen_circle = True
            while widen_circle:
                # look in a circle for cells
                bool_x0 = cell_properties.x <= xc + r
                bool_x1 = cell_properties.x >= xc - r
                bool_y0 = cell_properties.y <= yc + r
                bool_y1 = cell_properties.y >= yc - r
                bool_c = cell_properties.id != id
                # cell_properties_filtered = cell_properties.all(bool_x0, bool_x1, bool_y0, bool_y1)
                cell_properties_filtered = cell_properties[bool_x0 & bool_x1 & bool_y0 & bool_y1 & bool_c].copy()
                cell_properties_filtered['dist'] = ((cell_properties_filtered.x - xc)**2 + (cell_properties_filtered.y - yc)**2)**(1/2)
                cell_properties_filtered_circle = cell_properties_filtered[cell_properties_filtered.dist <= r].copy()
                if cell_properties_filtered_circle.shape[0] >= min_edges:
                    widen_circle = False
                else:
                    r += increase
            # Assign weighted edges for each cell within radius r
            for index_edge, row_edge in cell_properties_filtered_circle.iterrows():
                id_edge, dist = row_edge[['id','dist']]
                id_edge = int(id_edge)
                adjacency_matrix[id - 1, id_edge - 1] = dist
                adjacency_matrix[id_edge - 1, id - 1] = dist
    return(adjacency_matrix)


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
    parser.add_argument('-r', '--radius', dest = 'radius', type=int, default = 500, help='Output filename containing plots')
    parser.add_argument('-me', '--min_edges', dest = 'min_edges', type=int, default = 1, help='Output filename containing plots')
    # parser.add_argument('-gcm', '--graph_cmap', dest = 'graph_cmap', type=str, default = 'inferno', help='Output filename containing plots')
    # parser.add_argument('-ccm', '--cell_cmap', dest = 'cell_cmap', type=str, default = 'Blues', help='Output filename containing plots')
    # parser.add_argument('-vs', '--vertex_size', dest = 'vertex_size', type=int, default = 2, help='Output filename containing plots')
    # parser.add_argument('-vc', '--vertex_color', dest = 'vertex_color', type=str, default = 'w', help='Output filename containing plots')
    # parser.add_argument('-ew', '--edge_width', dest = 'edge_width', type=int, default = 2, help='Output filename containing plots')
    parser.add_argument('-s', '--save', dest = 'save', type=str, default= 'T', help='Output filename containing plots')
    parser.add_argument('-gext', '--graph_extension', dest = 'graph_extension', type=str, help='Output filename containing plots')
    # parser.add_argument('-gsext', '--graph_on_seg_extension', dest = 'graph_on_seg_extension', type=str, help='Output filename containing plots')
    parser.add_argument('-sext', '--seg_extension', dest = 'seg_extension', type=str, help='Output filename containing plots')
    parser.add_argument('-cpext', '--cell_props_extension', dest = 'cell_props_extension', type=str, help='Output filename containing plots')
    args = parser.parse_args()

    r = args.radius
    for seg_name in args.seg_names:
        # seg_filename = seg_name + args.seg_extension
        # seg = np.load(seg_filename).astype(int)
        cell_properties_filename = seg_name + args.cell_props_extension
        cell_properties = pd.read_csv(cell_properties_filename)
        cell_properties.sort_values('id', inplace = True)
        # Create an adjacency matrix with weighted edges
        adjacency_matrix = build_adjacency_matrix(cell_properties, r, args.min_edges)
        # adjacency_matrix = np.zeros([cell_properties.shape[0]]*2)
        # for index, row in cell_properties.iterrows():
        #     id, xc, yc = row[['id','x','y']]
        #     id = int(id)
        #     adj_row = adjacency_matrix[index,:]
        #     if  adj_row[id-1] == 0:
        #         # Make sure each cell is conneted to the graph
        #         widen_circle = True
        #         while widen_circle:
        #             # look in a circle for cells
        #             bool_x0 = cell_properties.x <= xc + r
        #             bool_x1 = cell_properties.x >= xc - r
        #             bool_y0 = cell_properties.y <= yc + r
        #             bool_y1 = cell_properties.y >= yc - r
        #             bool_c = cell_properties.id != id
        #             # cell_properties_filtered = cell_properties.all(bool_x0, bool_x1, bool_y0, bool_y1)
        #             cell_properties_filtered = cell_properties[bool_x0 & bool_x1 & bool_y0 & bool_y1 & bool_c].copy()
        #             cell_properties_filtered['dist'] = ((cell_properties_filtered.x - xc)**2 + (cell_properties_filtered.y - yc)**2)**(1/2)
        #             cell_properties_filtered_circle = cell_properties_filtered[cell_properties_filtered.dist <= r].copy()
        #             if cell_properties_filtered_circle.shape[0] > 0:
        #                 widen_circle = False
        #             else:
        #                 r += 50
        #         # Assign weighted edges for each cell within radius r
        #         for index_edge, row_edge in cell_properties_filtered_circle.iterrows():
        #             id_edge, dist = row_edge[['id','dist']]
        #             id_edge = int(id_edge)
        #             adjacency_matrix[id - 1, id_edge - 1] = dist
        #             adjacency_matrix[id_edge - 1, id - 1] = dist
        mean_neighbors = np.mean([np.sum(adjacency_matrix[i,:] > 0) for i in range(adjacency_matrix.shape[0])])
        print("mean_neighbors",mean_neighbors)
        if args.save == 'T':
            adjacency_matrix_filename = seg_name + args.graph_extension
            np.save(adjacency_matrix_filename, adjacency_matrix)
        else:
            plt.imshow(adjacency_matrix, cmap = 'inferno')
            plt.show()
            plt.close()

        # # Show graph on image
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


        # # plt.imshow(seg_color)
        # # plt.show()
        # # plt.close()
        # # edge_map = find_boundaries(seg, mode = 'inner')*seg
        # edge_map = filters.sobel(seg > 0)
        # print('edge_map', edge_map)
        # # plt.imshow(edge_map)
        # # plt.show()
        # # plt.close()
        # # rag = graph.rag_mean_color(seg_color, seg)
        # rag = graph.rag_boundary(seg, edge_map)
        # # edges_rgb = color.gray2rgb(edge_map)
        # # lc = graph.show_rag(seg, rag, edges_rgb, img_cmap='gray', edge_cmap='viridis', edge_width=1.2)
        # for i in rag.edges:
        #     for e in i:
        #         print(e)
        # out = graph.show_rag(seg, rag, seg_color)
        # io.show()
    return

if __name__ == '__main__':
    main()
