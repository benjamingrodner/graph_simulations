# Check out spot cell assignment and colocalization
import numpy as np
import matplotlib.pyplot as plt
import javabridge
import bioformats
import os
import re
from skimage import color
from skimage import restoration
from skimage.util import pad
import scipy.ndimage as ndimage
from skimage.measure import regionprops
from skimage.feature import peak_local_max
import javabridge
import bioformats
javabridge.start_vm(class_path=bioformats.JARS)


data_dir = 'data/images/airyscan/subset_05'
experiment_dir = 'experiments/2020_02_29_gfp_dna_redo'
# spot_props_coloc_filename = '2020_02_29_gfp_dna_redo_probeset_probesonly_plasmidfraction_1_fov_2_laser_561_spot_properties_cell_id_coloc.csv'
spot_props_coloc_filename = '2020_02_29_gfp_dna_redo_probeset_withblock_plasmidfraction_1_fov_4_laser_633_spot_properties_cell_id_coloc.csv'
spot_props_coloc_filename_full = '{}/{}/{}'.format(experiment_dir, data_dir, spot_props_coloc_filename)
spot_props_coloc = pd.read_csv(spot_props_coloc_filename_full)

# raw_filename = '2020_02_29_gfp_dna_redo_probeset_probesonly_plasmidfraction_1_fov_2_airyscan_processed.czi'
raw_filename = '2020_02_29_gfp_dna_redo_probeset_withblock_plasmidfraction_1_fov_4_airyscan_processed.czi'
raw_filename_full = '{}/{}/{}'.format(experiment_dir, data_dir, raw_filename)
raw = bioformats.load_image(raw_filename_full)
cell_raw = raw[:,:,0]
spot_raw = raw[:,:,2]
spot_561_raw = raw[:,:,1]

# spot_seg_filename = '2020_02_29_gfp_dna_redo_probeset_probesonly_plasmidfraction_1_fov_2_laser_561_seg.npy'
spot_seg_filename = '2020_02_29_gfp_dna_redo_probeset_withblock_plasmidfraction_1_fov_4_laser_633_seg.npy'
spot_seg_filename_full = '{}/{}/{}'.format(experiment_dir, data_dir, spot_seg_filename)
spot_seg = np.load(spot_seg_filename_full)
spots = regionprops(spot_seg, intensity_image=spot_raw)
spot_props = pd.DataFrame(columns = ['ID', 'X', 'Y', 'Area'])
for j in range(len(spots)):
    spot_props.loc[j, 'ID'] = spots[j].label
    spot_props.loc[j, 'X'] = spots[j].centroid[1]
    spot_props.loc[j, 'Y'] = spots[j].centroid[0]
    spot_props.loc[j, 'Area'] = spots[j].area
    spot_props.loc[j, 'Intensity'] = spots[j].mean_intensity

spot_561_seg_filename = '2020_02_29_gfp_dna_redo_probeset_withblock_plasmidfraction_1_fov_4_laser_561_seg.npy'
spot_561_seg_filename_full = '{}/{}/{}'.format(experiment_dir, data_dir, spot_561_seg_filename)
spot_561_seg = np.load(spot_561_seg_filename_full)
spots_561 = regionprops(spot_561_seg, intensity_image=spot_561_raw)
spot_561_props = pd.DataFrame(columns = ['ID', 'X', 'Y', 'Area'])
for j in range(len(spots)):
    spot_561_props.loc[j, 'ID'] = spots_561[j].label
    spot_561_props.loc[j, 'X'] = spots_561[j].centroid[1]
    spot_561_props.loc[j, 'Y'] = spots_561[j].centroid[0]
    spot_561_props.loc[j, 'Area'] = spots_561[j].area
    spot_561_props.loc[j, 'Intensity'] = spots_561[j].mean_intensity

# cell_props_filename = '2020_02_29_gfp_dna_redo_probeset_withblock_plasmidfraction_1_fov_4_avgint_norm.csv'
# cell_props_filename_full = '{}/{}/{}'.format(experiment_dir, data_dir, cell_props_filename)
# cell_props = pd.read_csv(cell_props_filename_full)
# cell_seg_filename = '2020_02_29_gfp_dna_redo_probeset_probesonly_plasmidfraction_1_fov_2_seg.npy'
cell_seg_filename = '2020_02_29_gfp_dna_redo_probeset_withblock_plasmidfraction_1_fov_4_seg.npy'
cell_seg_filename_full = '{}/{}/{}'.format(experiment_dir, data_dir, cell_seg_filename)
cell_seg = np.load(cell_seg_filename_full)
cells = regionprops(cell_seg, intensity_image=cell_raw)
cell_props = pd.DataFrame(columns = ['ID', 'X', 'Y', 'Area'])
for j in range(len(cells)):
    cell_props.loc[j, 'ID'] = cells[j].label
    cell_props.loc[j, 'X'] = cells[j].centroid[1]
    cell_props.loc[j, 'Y'] = cells[j].centroid[0]
    cell_props.loc[j, 'Area'] = cells[j].area
    cell_props.loc[j, 'Intensity'] = cells[j].mean_intensity

def sum_spot_intensity(df):
    int_spot_sum = df.Intensity * df.Area
    int_cell_sum =int_spot_sum.sum()
    return(int_cell_sum)


def normalized_spot_intensity(df, cell_props):
    # print('df id',df.ID.values[0])
    # print('df id isin cell props: ', any(cell_props.ID.isin(df.ID.values)))
    cell = cell_props[cell_props.ID == df.ID.values[0]]
    int_spot_sum = df.Intensity * df.Area
    int_cell_sum =int_spot_sum.sum() / cell.Area.values[0]
    return(int_cell_sum)
#     return(df.ID)
#
# x = spot_props_coloc.groupby('cell_id').apply(normalized_spot_intensity, cell_props=cell_props).values


def mean_spot_intensity(df):
    int_cell_mean = df.Intensity.sum()/df.shape[0]
    return(int_cell_mean)

def count_spots(df):
    spot_count = df.shape[0]
    return(spot_count)

def show_spot_props_hist(spot_props_coloc, cell_props, spot_seg, cell_seg):
    change_filter = 'y'
    while change_filter == 'y':
        print('Look at histogram? y/n')
        change_ylim = input()
        while change_ylim == 'y':
            print('Input ylim:')
            ylim = input()
            x = spot_props_coloc.loc[spot_props_coloc.cell_id != 0, 'dist']
            fig = plt.figure(frameon=False)
            fig.set_size_inches(20, 10)
            ax = fig.add_axes(0,0,1,1)
            ax.hist(x, color = (0,0.5,1), histtype = 'step', bins = 100)
            ax.title('Distance to cell in ' + spot_props_coloc_filename)
            # axes = plt.gca()
            ax.set_ylim([0, int(ylim)])
            plt.show()
            plt.close()
            print('Change ylim? y/n')
            change_ylim = input()
        print('Input distance filter: ')
        dist = float(input())
        fig = plt.figure(frameon=False)
        fig.set_size_inches(20, 10)
        spot_props_coloc_filtered = spot_props_coloc[spot_props_coloc.dist < dist]
        # Distance from spot to cell
        ax = fig.add_subplot(241)
        x = spot_props_coloc.loc[spot_props_coloc.cell_id != 0, 'dist']
        ax.hist(x, color = (0,0.5,1), histtype = 'step', bins = 100)
        x = np.tile([dist],20)
        y = list(range(20))
        ax.plot(x,y,color='red')
        ax.set_xlabel('Distance from spot to cell')
        # Number of spots in cells
        ax = fig.add_subplot(242)
        x = spot_props_coloc_filtered.loc[spot_props_coloc_filtered.cell_id != 0, 'cell_id'].value_counts().values
        ax.hist(x, color = (0,0.5,1), histtype = 'step', bins = 100)
        ax.set_xlabel('Number of dots assigned to cells')
        ax.set_title('File: ' + spot_props_coloc_filename)
        # Intensity of spots
        ax = fig.add_subplot(243)
        x = spot_props_coloc_filtered.Intensity.values
        ax.hist(x, color = (0,0.5,1), histtype = 'step', bins = 100)
        ax.set_xlabel('Spot intensity distribution')
        # Summed Intensity of spots per cell
        ax = fig.add_subplot(244)
        x = spot_props_coloc_filtered.groupby('cell_id').apply(sum_spot_intensity).values
        ax.hist(x, color = (0,0.5,1), histtype = 'step', bins = 100)
        ax.set_xlabel('Spot intensity summed for each cell')
        # # Normalized Intensity of spots per cell
        # ax = fig.add_subplot(244)
        # x = spot_props_coloc_filtered.groupby('cell_id').apply(normalized_spot_intensity, cell_props=cell_props).values
        # ax.hist(x, color = (0,0.5,1), histtype = 'step', bins = 100)
        # ax.set_xlabel('Spot intensity summed, normalized by cell area')
        # Mean Intensity of spots per cell
        ax = fig.add_subplot(245)
        x = spot_props_coloc_filtered.groupby('cell_id').apply(mean_spot_intensity).values
        ax.hist(x, color = (0,0.5,1), histtype = 'step', bins = 100)
        ax.set_xlabel('Spot intensity mean per cell')
        # Spot size
        ax = fig.add_subplot(246)
        x = spot_props_coloc_filtered.Area.values
        ax.hist(x, color = (0,0.5,1), histtype = 'step', bins = 100)
        ax.set_xlabel('Spot area in pixels')
        # Spot count and mean intensity 2d histogram
        ax = fig.add_subplot(247)
        x = spot_props_coloc_filtered.groupby('cell_id').apply(count_spots).values
        y = spot_props_coloc_filtered.groupby('cell_id').apply(mean_spot_intensity).values
        ax.hist2d(x, y, cmap = 'inferno', bins = [200,200])
        ax.set_xlim(0,20)
        ax.set_xlabel('Spot count per cell')
        ax.set_ylabel('mean spot intensity for cell')
        # Image
        ax = fig.add_subplot(248)
        sp_remove = spot_props_coloc.loc[spot_props_coloc.dist >= dist, 'cell_id'].values
        # spot_seg_filtered = np.isin(spot_seg, sp_remove)
        spot_seg_mask = ~np.isin(spot_seg, sp_remove)
        spot_seg_filtered = spot_seg * spot_seg_mask
        spot_seg_rgb = color.label2rgb(spot_seg_filtered, colors=[(0.7,0.1,0.7)], bg_label=0, bg_color=(0, 0, 0))
        cell_seg_rgb = color.label2rgb(cell_seg, colors=[(0.1,0.7,0.1)], bg_label=0, bg_color=(0, 0, 0))
        merge = spot_seg_rgb + cell_seg_rgb
        # merge = spot_seg_rgb
        ax.imshow(merge)
        xlims = [810, 1110]
        ylims = [1110, 810]
        spot_props_reduced = spot_props[(spot_props.X > xlims[0]) & (spot_props.X < xlims[1]) & (spot_props.Y < ylims[0]) & (spot_props.Y > ylims[1])]
        cell_props_reduced = cell_props[(cell_props.X > xlims[0]) & (cell_props.X < xlims[1]) & (cell_props.Y < ylims[0]) & (cell_props.Y > ylims[1])]
        for i in range(spot_props.shape[0]):
            spot = spot_props.iloc[i,:]
            x = spot['X']
            y = spot['Y']
            lab = str(int(spot['ID'])) + ', ' + str(spot['dist'])
            plt.text(x,y,lab, ha="center", va="center", color='white')
        ax.set_xlim(xlims[0],xlims[1])
        ax.set_ylim(ylims[0],ylims[1])
        ax.set_xlabel('Segmented image merge')
        # Final
        plt.title('File: ' + spot_props_coloc_filename)
        plt.savefig('{}/figures/spot_hist_zoom_2020_02_29_gfp_dna_redo_probeset_withblock_plasmidfraction_1_fov_4_airyscan_processed.png'.format(experiment_dir))
        plt.close()
        print('Change distance filter? y/n')
        change_filter = input()
    return

show_spot_props_hist(spot_props_coloc, cell_props, spot_seg, cell_seg)

def show_spot_cell_association(cell_raw, spot_raw, spot_props_coloc, cell_props, spot_seg, cell_seg):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(20, 10)
    # xlims = [740, 1110]
    # ylims = [810, 1110]
    xlims = [0, cell_raw.shape[1]]
    ylims = [0, cell_raw.shape[0]]
    spot_props_reduced = spot_props[(spot_props.X > xlims[0]) & (spot_props.X < xlims[1]) & (spot_props.Y > ylims[0]) & (spot_props.Y < ylims[1])]
    # spot_props_reduced = spot_props[(spot_props.X > xlims[0]) & (spot_props.X < xlims[1]) & (spot_props.Y > ylims[0]) & (spot_props.Y < ylims[1])]
    cell_props_reduced = cell_props[(cell_props.X > xlims[0]) & (cell_props.X < xlims[1]) & (cell_props.Y > ylims[0]) & (cell_props.Y < ylims[1])]
    # cell_props_reduced = cell_props[(cell_props.X > xlims[0]) & (cell_props.X < xlims[1]) & (cell_props.Y > ylims[0]) & (cell_props.Y < ylims[1])]
    # spot_props_reduced['X'] = spot_props_reduced['Y'] - 740
    # spot_props_reduced['Y'] = spot_props_reduced['Y'] - 810
    # Raw cell
    ax = fig.add_subplot(231)
    ax.imshow(cell_raw, cmap='inferno')
    ax.set_xlabel('Raw gfp')
    ax.set_xlim(xlims[0],xlims[1])
    ax.set_ylim(ylims[0],ylims[1])
    # Raw spot
    ax = fig.add_subplot(232)
    ax.imshow(spot_raw, cmap='inferno')
    ax.set_xlabel('Raw spot')
    ax.set_xlim(xlims[0],xlims[1])
    ax.set_ylim(ylims[0],ylims[1])
    # Spot seg id labeled
    # ax = fig.add_axes([0,0,1,1])
    ax = fig.add_subplot(233)
    spot_seg_rgb = color.label2rgb(spot_seg, bg_label=0, bg_color=(0, 0, 0))
    # spot_seg_rgb = color.label2rgb(spot_seg, colors=[(0.7,0.1,0.7)], bg_label=0, bg_color=(0, 0, 0))
    ax.imshow(spot_seg_rgb)
    for i in range(spot_props_reduced.shape[0]):
        spot = spot_props_reduced.iloc[i,:]
        x = spot['X']
        y = spot['Y']
        lab = int(spot['cell_id'])
        ax.text(x,y,lab, ha="center", va="center", color='white')
    # ax.text(800,900, 'Here!', ha="center", va="center", color='white')
    ax.set_xlabel('Spot seg with spot ID')
    ax.set_xlim(xlims[0],xlims[1])
    ax.set_ylim(ylims[0],ylims[1])
    # Cell seg id labeled
    ax = fig.add_subplot(234)
    cell_seg_rgb = color.label2rgb(cell_seg, colors=[(0.1,0.7,0.1)], bg_label=0, bg_color=(0, 0, 0))
    ax.imshow(cell_seg_rgb)
    for i in range(cell_props_reduced.shape[0]):
        cell = cell_props_reduced.iloc[i,:]
        y = cell['X']
        x = cell['Y']
        lab = int(cell['ID'])
        ax.text(x,y,lab, ha="center", va="center", color='white')
    ax.text(850,950, 'HERE!', ha="center", va="center", color='white')
    ax.set_xlabel('Cell seg with spot ID')
    ax.set_xlim(xlims[0],xlims[1])
    ax.set_ylim(ylims[0],ylims[1])
    # Spot with cell id
    ax = fig.add_subplot(235)
    spot_seg_rgb = color.label2rgb(spot_seg, colors=[(0.7,0.1,0.7)], bg_label=0, bg_color=(0, 0, 0))
    cell_seg_rgb = color.label2rgb(cell_seg, colors=[(0.1,0.7,0.1)], bg_label=0, bg_color=(0, 0, 0))
    merge = spot_seg_rgb + cell_seg_rgb
    ax.imshow(merge)
    for i in range(cell_props_reduced.shape[0]):
        cell = cell_props_reduced.iloc[i,:]
        y = cell['X']
        x = cell['Y']
        lab = int(cell['ID'])
        ax.text(x,y,lab, ha="center", va="center", color='white')
    for i in range(spot_props_reduced.shape[0]):
        spot = spot_props_reduced.iloc[i,:]
        y = spot['X']
        x = spot['Y']
        lab = str(int(spot['cell_id'])) + ', ' + str(spot['dist'])
        ax.text(x,y,lab, ha="center", va="center", color='white')
    ax.set_xlabel('Spot assignemnt to cell id with distance')
    ax.set_xlim(xlims[0],xlims[1])
    ax.set_ylim(ylims[0],ylims[1])
    # show
    # plt.savefig('{}/figures/spot_association_zoom_2020_02_29_gfp_dna_redo_probeset_withblock_plasmidfraction_1_fov_4_airyscan_processed.png'.format(experiment_dir))
    plt.show()
    plt.title('Sample ' + raw_filename)
    plt.close()

show_spot_cell_association(cell_raw, spot_raw, spot_props_coloc, cell_props, spot_seg, cell_seg)

# Test skimage morphology
ylims = [540, 440]
xlims = [700,800]
cell_seg_zoom = cell_seg[ylims[1]:ylims[0],xlims[0]:xlims[1]]
cell_seg_zoom[20:30,60:70]
x4 = cell_props.loc[cell_props.ID == 4, 'X'].values[0] - 440
y4 = cell_props.loc[cell_props.ID == 4, 'Y'].values[0] - 700
x5 = cell_props.loc[cell_props.ID == 5, 'X'].values[0] - 440
y5 = cell_props.loc[cell_props.ID == 5, 'Y'].values[0] - 700
cell_seg_zoom_rgb = color.label2rgb(cell_seg_zoom, bg_label=0, bg_color=(0, 0, 0))
plt.imshow(cell_seg_zoom_rgb)
plt.text(y4,x4,'4', ha="center", va="center", color='white')
plt.text(y5,x5,'5', ha="center", va="center", color='white')
plt.show()
plt.close()


# Test local max finder

# 2628, 2629
def remove_close_maxima(df, close_maxima_filter):
    spot_local_max_filtered = df
    i = 0
    while i < spot_local_max_filtered.shape[0]:
        rowi = spot_local_max_filtered.iloc[i, :]
        index = rowi.name
        for jndex, rowj in spot_local_max_filtered[spot_local_max_filtered.index != index].iterrows():
            xj, yj = rowj[['max_x','max_y']]
            xi, yi = rowi[['max_x', 'max_y']]
            dist = ((xj-xi)**2 + (yj-yi)**2)**(1/2)
            if dist <= close_maxima_filter:
                spot_local_max_filtered = spot_local_max_filtered.drop(jndex, axis=0)
            if int(rowi['max_id']) == 2628:
                print('df',df)
                print('xi,xj',xi, xj)
                print('yi,yj',yi, yj)
                print('dist',dist)
        i += 1
    return(spot_local_max_filtered)

def find_spot_local_max(spot_raw, spot_seg, min_distance, close_maxima_filter):
    spot_raw_masked = spot_raw * (spot_seg > 0)
    spot_local_max = peak_local_max(spot_raw_masked, min_distance=min_distance)
    spot_local_max_df = pd.DataFrame()
    spot_local_max_temp = pd.DataFrame()
    for i in range(spot_local_max.shape[0]):
        xi = int(spot_local_max[i, 1])
        yi = int(spot_local_max[i, 0])
        spot_id = spot_seg[yi, xi]
        spot_local_max_temp = spot_local_max_temp.append({'spot_id':spot_id, 'max_id':i, 'max_x': xi, 'max_y': yi}, ignore_index=True)
    spot_local_max_df = spot_local_max_temp.groupby('spot_id').apply(remove_close_maxima, close_maxima_filter)
    return(spot_local_max_df)

min_distance = 5
close_maxima_filter = 5
spot_local_maxima = find_spot_local_max(spot_raw, spot_seg, min_distance, close_maxima_filter)
spot_561_local_maxima = find_spot_local_max(spot_561_raw, spot_561_seg, min_distance, close_maxima_filter)

def get_spot_max_cell_ids(cells_seg, spot_local_max, r, downsample):
    id_df = pd.DataFrame()
    cells_seg_pad = pad(cells_seg, r, mode='edge')
    for i in range(spot_local_max.shape[0]):
        print('Spot {} of {}'.format(i+1, spot_local_max.shape[0]), end='\r')
        # xi, yi = int(spot_props.loc[i, ['X','Y']])
        # xi = int(spot_local_max.loc[i, 'max_x'])
        # yi = int(spot_local_max.loc[i, 'max_y'])
        xi = int(spot_local_max.max_x.values[i] + r)
        yi = int(spot_local_max.max_y.values[i] + r)
        # Get the cell id for the centroid of the spot
        cell_id = cells_seg_pad[yi,xi]
        dist = 0
        # If the spot is not in a cell, check nearby
        nearby_df = pd.DataFrame()
        if cell_id == 0:
            # Check in a circle around the centroid
            for xr in np.arange(xi - r, xi + r + downsample, downsample):
            # for xr in range(xi-r,xi+r):
                for yr in np.arange(yi - r, yi + r + downsample, downsample):
                # for yr in range(yi-r,yi+r):
                    if (xi - xr)**2 + (yi - yr)**2 <= r**2:
                        nearby_cell = cells_seg_pad[yr,xr]
                        if not nearby_cell == 0:
                            dist = ((xr-xi)**2 + (yr-yi)**2)**(1/2)
                            nearby_df = nearby_df.append({'nearby_cell':nearby_cell, 'yr':yr, 'xr':xr,'dist':dist}, ignore_index=True)
            # Select nearest cell and return distance
            if not nearby_df.shape[0] == 0:
                cell_id = nearby_df.loc[nearby_df.dist == nearby_df.dist.min(), 'nearby_cell'].values[0]
                dist = nearby_df.dist.min()
        # Add the end cell id and distance to the temp dataframe
        id_df = id_df.append({'max_id':spot_local_max.max_id.values[i], 'cell_id':cell_id, 'dist':dist}, ignore_index=True)
    print('\n')
    return(id_df)

r = 50
downsample = 4
# spot_raw_masked = spot_raw * (spot_seg > 0)
# spot_561_raw_masked = spot_561_raw * (spot_561_seg > 0)
# spot_local_maxima = peak_local_max(spot_raw_masked, min_distance=min_distance)
# spot_561_local_maxima = peak_local_max(spot_561_raw_masked, min_distance=min_distance)
ylims = [550, 350]
xlims = [650,850]
cell_seg_reduced = cell_seg[ylims[1]:ylims[0],xlims[0]:xlims[1]]
cell_raw_reduced = cell_raw[ylims[1]:ylims[0],xlims[0]:xlims[1]]
spot_seg_reduced = spot_seg[ylims[1]:ylims[0],xlims[0]:xlims[1]]
spot_raw_reduced = spot_raw[ylims[1]:ylims[0],xlims[0]:xlims[1]]
spot_561_seg_reduced = spot_561_seg[ylims[1]:ylims[0],xlims[0]:xlims[1]]
spot_561_raw_reduced = spot_561_raw[ylims[1]:ylims[0],xlims[0]:xlims[1]]
cell_props_reduced = cell_props[(cell_props.X > xlims[0]) & (cell_props.X < xlims[1]) & (cell_props.Y < ylims[0]) & (cell_props.Y > ylims[1])]
cell_props_reduced = cell_props_reduced.rename(columns={'X':'X_old', 'Y':'Y_old'})
cell_props_reduced['X'] = cell_props_reduced['X_old'] - xlims[0]
cell_props_reduced['Y'] = cell_props_reduced['Y_old'] - ylims[1]
spot_props_reduced = spot_props[(spot_props.X > xlims[0]) & (spot_props.X < xlims[1]) & (spot_props.Y < ylims[0]) & (spot_props.Y > ylims[1])]
spot_props_reduced = spot_props_reduced.rename(columns={'X':'X_old', 'Y':'Y_old'})
spot_props_reduced['X'] = spot_props_reduced['X_old'] - xlims[0]
spot_props_reduced['Y'] = spot_props_reduced['Y_old'] - ylims[1]
spot_561_props_reduced = spot_561_props[(spot_561_props.X > xlims[0]) & (spot_561_props.X < xlims[1]) & (spot_561_props.Y < ylims[0]) & (spot_561_props.Y > ylims[1])]
spot_561_props_reduced = spot_561_props_reduced.rename(columns={'X':'X_old', 'Y':'Y_old'})
spot_561_props_reduced['X'] = spot_561_props_reduced['X_old'] - xlims[0]
spot_561_props_reduced['Y'] = spot_561_props_reduced['Y_old'] - ylims[1]
spot_local_maxima_reduced = spot_local_maxima.loc[(spot_local_maxima.loc[:,'max_x'] > xlims[0]) & (spot_local_maxima.loc[:,'max_x'] < xlims[1]) & (spot_local_maxima.loc[:,'max_y'] < ylims[0]) & (spot_local_maxima.loc[:,'max_y'] > ylims[1]),:]
spot_local_maxima_reduced = spot_local_maxima_reduced.rename(columns={'max_x':'max_x_old', 'max_y':'max_y_old'})
spot_local_maxima_reduced['max_x'] = spot_local_maxima_reduced['max_x_old'] - xlims[0]
spot_local_maxima_reduced['max_y'] = spot_local_maxima_reduced['max_y_old'] - ylims[1]
spot_561_local_maxima_reduced = spot_561_local_maxima.loc[(spot_561_local_maxima.loc[:,'max_x'] > xlims[0]) & (spot_561_local_maxima.loc[:,'max_x'] < xlims[1]) & (spot_561_local_maxima.loc[:,'max_y'] < ylims[0]) & (spot_561_local_maxima.loc[:,'max_y'] > ylims[1]),:]
spot_561_local_maxima_reduced = spot_561_local_maxima_reduced.rename(columns={'max_x':'max_x_old', 'max_y':'max_y_old'})
spot_561_local_maxima_reduced['max_x'] = spot_561_local_maxima_reduced['max_x_old'] - xlims[0]
spot_561_local_maxima_reduced['max_y'] = spot_561_local_maxima_reduced['max_y_old'] - ylims[1]
# spot_max_cell_id = get_spot_max_cell_ids(cell_seg, spot_local_maxima, r, downsample)
# spot_561_max_cell_id = get_spot_max_cell_ids(cell_seg, spot_561_local_maxima, r, downsample)
# spot_max_cell_id_merge = spot_local_maxima.merge(spot_max_cell_id, on='max_id', how='left')
# spot_561_max_cell_id_merge = spot_561_local_maxima.merge(spot_561_max_cell_id, on='max_id', how='left')
spot_max_cell_id_reduced = get_spot_max_cell_ids(cell_seg_reduced, spot_local_maxima_reduced, r, downsample)
spot_561_max_cell_id_reduced = get_spot_max_cell_ids(cell_seg_reduced, spot_561_local_maxima_reduced, r, downsample)
spot_max_cell_id_reduced_merge = spot_local_maxima_reduced.merge(spot_max_cell_id_reduced, on='max_id', how='left')
spot_561_max_cell_id_reduced_merge = spot_561_local_maxima_reduced.merge(spot_561_max_cell_id_reduced, on='max_id', how='left')

def colocalize_two_color_spots(spot_props_list, spot_seg_list, r, downsample):
    spot_props_coloc_list = []
    # Get reverse index for comparing colors
    other_index_list = list(reversed(range(len(spot_props_list))))
    for i in range(len(spot_props_list)):
        # print('i: ',i)
        other_index = other_index_list[i]
        # spots to analyze
        spot_props = spot_props_list[i]
        # Other colored spots
        spot_seg_other = spot_seg_list[other_index]
        spot_seg_other_pad = pad(spot_seg_other, r, mode='edge')
        # print('spot_seg_other_pad.shape: ', spot_seg_other_pad.shape)
        coloc_df = pd.DataFrame()
        for j in range(spot_props.shape[0]):
            print('Spot {} of {}'.format(j+1, spot_props.shape[0]), end='\r')
            # xj, yj = int(spot_props.loc[j, ['X','Y']])
            # xj = int(spot_props.loc[j, 'X'])
            # yj = int(spot_props.loc[j, 'Y'])
            xj = int(spot_props.max_x.values[j] + r)
            yj = int(spot_props.max_y.values[j] + r)
            # Get the cell id for the centroid of the spot
            spot_other_id = spot_seg_other_pad[yj,xj]
            dist = 0
            # If the spot is not in a cell, check nearby
            nearby_df = pd.DataFrame()
            if spot_other_id == 0:
                # Check in a circle around the centroid
                for xr in np.arange(xj - r,xj + r + downsample, downsample):
                # for xr in range(xj-r,xj+r):
                    for yr in np.arange(yj - r, yj + r + downsample, downsample):
                    # for yr in range(yj-r,yj+r):
                        if (xj - xr)**2 + (yj - yr)**2 <= r**2:
                            nearby_spot = spot_seg_other_pad[yr,xr]
                            if not nearby_spot == 0:
                                dist = ((xj-xr)**2 + (yj-yr)**2)**(1/2)
                                nearby_df = nearby_df.append({'nearby_spot':nearby_spot, 'yr':yr, 'xr':xr,'dist':dist}, ignore_index=True)
                # Select nearest cell and return distance
                if not nearby_df.shape[0] == 0:
                    spot_other_id = nearby_df.loc[nearby_df.dist == nearby_df.dist.min(), 'nearby_spot'].values[0]
                    dist = nearby_df.dist.min()
            # Add the end cell id and distance to the temp dataframe
            coloc_df = coloc_df.append({'max_id':spot_props.max_id.values[j], 'spot_other_id':spot_other_id, 'dist_spot':dist}, ignore_index=True)
        # Merge the cell ids with the spot properties dataframe
        # spot_props_coloc = spot_props.merge(coloc_df, how='left', on='ID')
        # spot_props_coloc_list.append(spot_props_coloc)
        spot_props_coloc_list.append(coloc_df)
        print('\n')
    return(spot_props_coloc_list)

spot_props_reduced_list = [spot_561_max_cell_id_reduced_merge, spot_max_cell_id_reduced_merge]
spot_seg_reduced_list = [spot_561_seg_reduced, spot_seg_reduced]
r = 50
downsample = 4
spot_max_coloc_reduced_list = colocalize_two_color_spots(spot_props_reduced_list, spot_seg_reduced_list, r, downsample)
spot_max_cell_id_coloc_reduced_merge = spot_max_cell_id_reduced_merge.merge(spot_max_coloc_reduced_list[1], on='max_id', how='left')
spot_561_max_cell_id_coloc_reduced_merge = spot_561_max_cell_id_reduced_merge.merge(spot_max_coloc_reduced_list[0], on='max_id', how='left')

def show_zoom_cell_and_spot_with_id(save, out_filename, ft, col, xlims, ylims, raw_filename, cell_raw, spot_raw, spot_561_raw, cell_props, spot_props, spot_561_props, cell_seg, spot_seg, spot_561_seg, spot_local_maxima, spot_561_local_maxima):
    cell_props_reduced = cell_props[(cell_props.X > xlims[0]) & (cell_props.X < xlims[1]) & (cell_props.Y < ylims[0]) & (cell_props.Y > ylims[1])]
    spot_props_reduced = spot_props[(spot_props.X > xlims[0]) & (spot_props.X < xlims[1]) & (spot_props.Y < ylims[0]) & (spot_props.Y > ylims[1])]
    spot_561_props_reduced = spot_561_props[(spot_561_props.X > xlims[0]) & (spot_561_props.X < xlims[1]) & (spot_561_props.Y < ylims[0]) & (spot_561_props.Y > ylims[1])]
    spot_local_maxima_reduced = spot_local_maxima.loc[(spot_local_maxima.loc[:,'max_x'] > xlims[0]) & (spot_local_maxima.loc[:,'max_x'] < xlims[1]) & (spot_local_maxima.loc[:,'max_y'] < ylims[0]) & (spot_local_maxima.loc[:,'max_y'] > ylims[1]),:]
    # spot_local_maxima_reduced = spot_local_maxima[(spot_local_maxima[:,1] > xlims[0]) & (spot_local_maxima[:,1] < xlims[1]) & (spot_local_maxima[:,0] < ylims[0]) & (spot_local_maxima[:,0] > ylims[1]),:]
    spot_561_local_maxima_reduced = spot_561_local_maxima.loc[(spot_561_local_maxima.loc[:,'max_x'] > xlims[0]) & (spot_561_local_maxima.loc[:,'max_x'] < xlims[1]) & (spot_561_local_maxima.loc[:,'max_y'] < ylims[0]) & (spot_561_local_maxima.loc[:,'max_y'] > ylims[1]),:]
    # spot_561_local_maxima_reduced = spot_561_local_maxima[(spot_561_local_maxima[:,1] > xlims[0]) & (spot_561_local_maxima[:,1] < xlims[1]) & (spot_561_local_maxima[:,0] < ylims[0]) & (spot_561_local_maxima[:,0] > ylims[1]),:]
    # spot_max_cell_id_reduced = spot_max_cell_id.loc[(spot_max_cell_id.loc[:,'max_x'] > xlims[0]) & (spot_max_cell_id.loc[:,'max_x'] < xlims[1]) & (spot_max_cell_id.loc[:,'max_y'] < ylims[0]) & (spot_max_cell_id.loc[:,'max_y'] > ylims[1]),:]
    # spot_561_max_cell_id_reduced = spot_561_max_cell_id.loc[(spot_561_max_cell_id.loc[:,'max_x'] > xlims[0]) & (spot_561_max_cell_id.loc[:,'max_x'] < xlims[1]) & (spot_561_max_cell_id.loc[:,'max_y'] < ylims[0]) & (spot_561_max_cell_id.loc[:,'max_y'] > ylims[1]),:]
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ft, ft)
    plt.rcParams["text.color"] = col
    plt.rcParams["axes.labelcolor"] = col
    plt.rcParams["xtick.color"] =  col
    plt.rcParams["ytick.color"] = col
    plt.rcParams['lines.color'] = col
    plt.rcParams['font.size'] = ft
    plt.title('Sample: ' + raw_filename, y=1.1)
    plt.axis('off')
    #
    ax = fig.add_subplot(231)
    im = ax.imshow(cell_raw, cmap='inferno')
    ax.set_title('Raw cell', y=1.05)
    ax.set_xlim(xlims[0],xlims[1])
    ax.set_ylim(ylims[0],ylims[1])
    cax = fig.add_axes([0.15, 0.5, 0.2, 0.02])
    fig.colorbar(im, cax=cax, orientation='horizontal')
    #
    ax = fig.add_subplot(234)
    print('cell_seg.shape', cell_seg.shape)
    cell_seg_rgb = color.label2rgb(cell_seg, colors=[(0.1,0.9,0.1)], bg_label=0, bg_color=(0, 0, 0))
    # spot_seg_zoom_rgb = color.label2rgb(spot_seg_zoom, bg_label=0, bg_color=(0, 0, 0))
    ax.imshow(cell_seg_rgb)
    ax.set_title('Segmented Cell', y=1.05)
    for i in range(cell_props_reduced.shape[0]):
        cell = cell_props_reduced.iloc[i,:]
        x = cell['X']
        y = cell['Y']
        lab = int(cell['ID'])
        ax.text(x,y,lab, ha="center", va="center", color='gray')
    # ax.plot(np.linspace(xlims[0]+25,xlims[0]+35,10), np.repeat(ylims[1]+25, 10), 'w', linewidth=ft/4)
    # plt.imshow(spot_seg_zoom_rgb)
    # plt.text(x37,y37,'37', ha="center", va="center", color='white')
    # plt.text(y5,x5,'5', ha="center", va="center", color='white')
    ax.set_xlim(xlims[0],xlims[1])
    ax.set_ylim(ylims[0],ylims[1])
    #
    ax = fig.add_subplot(232)
    im = ax.imshow(spot_561_raw, cmap='inferno')
    ax.set_title('Raw Spot: 561 laser', y=1.05)
    ax.set_xlim(xlims[0],xlims[1])
    ax.set_ylim(ylims[0],ylims[1])
    cax = fig.add_axes([0.15 + 0.28, 0.5, 0.2, 0.02])
    fig.colorbar(im, cax=cax, orientation='horizontal')
    #
    ax = fig.add_subplot(235)
    spot_561_seg_rgb = color.label2rgb(spot_561_seg, colors=[(0.1,0.1,0.9)], bg_label=0, bg_color=(0, 0, 0))
    # spot_seg_zoom_rgb = color.label2rgb(spot_seg_zoom, bg_label=0, bg_color=(0, 0, 0))
    ax.imshow(spot_561_seg_rgb)
    ax.set_title('Setgmented Spot: 561 laser', y=1.05)
    for i in range(spot_561_local_maxima_reduced.shape[0]):
        spot = spot_561_local_maxima_reduced.iloc[i,:]
        x = spot['max_x']
        y = spot['max_y']
        lab = str(int(spot['max_id'])) + ', ' + str(int(spot['spot_other_id'])) + ', ' + str(spot['dist_spot'])
        # lab = str(int(spot['cell_id'])) + ', ' + str(int(spot['spot_id'])) + ', ' + str(spot['dist'])
        ax.text(x,y,lab, ha="left", va="baseline", color='white', size=ft*2/3)
    for i in range(spot_561_props_reduced.shape[0]):
        spot = spot_561_props_reduced.iloc[i,:]
        x = spot['X']
        y = spot['Y']
        lab = int(spot['ID'])
        ax.text(x,y,lab, ha="center", va="center", color='gray')
    ax.plot(spot_561_local_maxima_reduced.loc[:,'max_x'], spot_561_local_maxima_reduced.loc[:,'max_y'], 'w.', markersize=ft)
    # ax.plot(spot_561_local_maxima_reduced[:,1], spot_561_local_maxima_reduced[:,0], 'w.', markersize=ft)
    # plt.imshow(spot_seg_zoom_rgb)
    # plt.text(x37,y37,'37', ha="center", va="center", color='white')
    # plt.text(y5,x5,'5', ha="center", va="center", color='white')
    ax.set_xlim(xlims[0],xlims[1])
    ax.set_ylim(ylims[0],ylims[1])
    #
    ax = fig.add_subplot(233)
    im = ax.imshow(spot_raw, cmap='inferno')
    ax.set_title('Raw Spot: 633 laser', y=1.05)
    ax.set_xlim(xlims[0],xlims[1])
    ax.set_ylim(ylims[0],ylims[1])
    cax = fig.add_axes([0.15 + 0.28*2, 0.5, 0.2, 0.02])
    fig.colorbar(im, cax=cax, orientation='horizontal')
    #
    ax = fig.add_subplot(236)
    spot_seg_rgb = color.label2rgb(spot_seg, colors=[(0.9,0.1,0.1)], bg_label=0, bg_color=(0, 0, 0))
    # spot_seg_zoom_rgb = color.label2rgb(spot_seg_zoom, bg_label=0, bg_color=(0, 0, 0))
    ax.imshow(spot_seg_rgb)
    ax.set_title('Setgmented Spot: 633 laser', y=1.05)
    for i in range(spot_local_maxima_reduced.shape[0]):
        spot = spot_local_maxima_reduced.iloc[i,:]
        x = spot['max_x']
        y = spot['max_y']
        # lab = str(int(spot['cell_id'])) + ', ' + str(spot['dist'])
        lab = str(int(spot['max_id'])) + ', ' + str(int(spot['spot_other_id'])) + ', ' + str(spot['dist_spot'])
        # lab = str(int(spot['cell_id'])) + ', ' + str(int(spot['spot_id'])) + ', ' + str(spot['dist'])
        ax.text(x,y,lab, ha="left", va="baseline", color='white', size=ft*2/3)
    for i in range(spot_props_reduced.shape[0]):
        spot = spot_props_reduced.iloc[i,:]
        x = spot['X']
        y = spot['Y']
        lab = int(spot['ID'])
        ax.text(x,y,lab, ha="center", va="center", color='gray')
    ax.plot(spot_local_maxima_reduced.loc[:,'max_x'], spot_local_maxima_reduced.loc[:,'max_y'], 'w.', markersize=ft*2/3)
    # ax.plot(spot_local_maxima_reduced[:,1], spot_local_maxima_reduced[:,0], 'w.', markersize=ft)
    # plt.imshow(spot_seg_zoom_rgb)
    # plt.text(x37,y37,'37', ha="center", va="center", color='white')
    # plt.text(y5,x5,'5', ha="center", va="center", color='white')
    ax.set_xlim(xlims[0],xlims[1])
    ax.set_ylim(ylims[0],ylims[1])
    plt.subplots_adjust(hspace=0.65)
    if save == 'T':
        plt.savefig(out_filename, bbox_inches='tight', transparent=True)
    else:
        plt.show()
    plt.close()
    return

out_filename = '{}/figures/cell_spot_seg_comparison_2020_02_29_gfp_dna_redo_probeset_withblock_plasmidfraction_1_fov_4_airyscan_processed.png'.format(experiment_dir)
save = 'F'
# ylims = [550, 350]
# xlims = [650,850]
ylims = [cell_seg_reduced.shape[0], 0]
xlims = [0,cell_seg_reduced.shape[1]]
col = 'black'
ft = 20
show_zoom_cell_and_spot_with_id(save, out_filename, ft, col, xlims, ylims, raw_filename, cell_raw_reduced,
                                spot_raw_reduced, spot_561_raw_reduced, cell_props_reduced, spot_props_reduced,
                                spot_561_props_reduced, cell_seg_reduced, spot_seg_reduced, spot_561_seg_reduced,
                                spot_max_cell_id_coloc_reduced_merge, spot_561_max_cell_id_coloc_reduced_merge)
# show_zoom_cell_and_spot_with_id(save, out_filename, ft, col, xlims, ylims, raw_filename, cell_raw, spot_raw, spot_561_raw, cell_props, spot_props, spot_561_props, cell_seg, spot_seg, spot_561_seg, spot_max_cell_id_merge, spot_561_max_cell_id_merge)


# Remove dots
filename = '/workdir/bmg224/hiprfish/mobile_elements/experiments/2020_02_29_gfp_dna_redo/data/images/airyscan/subset_04/2020_02_29_gfp_dna_redo_probeset_withblock_plasmidfraction_1_fov_4_laser_561_seg.npy'
# filename = '/workdir/bmg224/hiprfish/mobile_elements/experiments/2020_02_29_gfp_dna_redo/data/images/airyscan/subset_04/2020_02_29_gfp_dna_redo_probeset_withblock_plasmidfraction_0_fov_2_seg.npy'
# filename = '/workdir/bmg224/hiprfish/mobile_elements/experiments/2020_02_29_gfp_dna_redo/data/images/airyscan/subset_04/2020_02_29_gfp_dna_redo_probeset_probesonly_plasmidfraction_0_fov_2_seg.npy'
raw_filename = re.sub('laser_561_seg.npy','airyscan_processed.czi',filename)
raw = bioformats.load_image(raw_filename)[:,:,1]
image = np.load(filename)
image_color =  color.label2rgb(image, bg_label=0, bg_color=(0,0,0))
image_filter = remove_small_objects(image, 20)
image_filter_color = color.label2rgb(image_filter, bg_label=0, bg_color=(0,0,0))
fig = plt.figure()
fig.set_size_inches(20,10)
ax = fig.add_subplot(131)
raw_log = np.log10(raw)
ax.imshow(raw_log, cmap='inferno')
ax = fig.add_subplot(132)
immean = np.mean(raw)
steepness = 100
shift = immean*steepness
raw_enh = raw**2
# raw_enh = 1/(1 + np.exp(-steepness*raw + shift))
ax.imshow(raw_enh, cmap = 'inferno')
# ax.imshow(image_color)
ax = fig.add_subplot(133)
ax.imshow(image_filter_color)
plt.show()
plt.close()
