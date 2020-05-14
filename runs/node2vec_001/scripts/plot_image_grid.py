# Create a gel-like plot for 2019_10_12_saber_gfp experiment data
# Ben Grodner

import pandas as pd
import numpy as np
import argparse
from scipy import stats
import glob
import re
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

################################################################################################################################################################
# Functions
################################################################################################################################################################


# def set_plot_params(ft, x, xlabel, xticklabels, y, ylabel, color):
def set_plot_params(ft, xlabel, xticks, xticklabels, ylabel, color, aspect, bottom, xl_coords, yl_coords, xl_font_scaling, y_axis_lim):
    fig, ax = plt.subplots(1, figsize=(ft*1.5,ft*1.5))
    # fig, ax = plt.subplots(1, figsize=(18 * 2, 18))
    ax.set_xlabel(xlabel, labelpad=500)
    ax.set_ylabel(ylabel)
    ax.spines['bottom'].set_color(color)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_color(color)
    ax.spines['right'].set_visible(False)
    for axis in ['bottom', 'top', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.15*ft)
    ax.xaxis.label.set_color(color)
    ax.yaxis.label.set_color(color)
    ax.xaxis.label.set_fontsize(xl_font_scaling*ft)
    ax.yaxis.label.set_fontsize(ft)
    ax.yaxis.label.set_rotation(0)
    # ax.xaxis.set_label_coords(-0.3, -0.02)
    ax.xaxis.set_label_coords(xl_coords[0], xl_coords[1])
    # ax.yaxis.set_label_coords(-0.25, 0.5)
    ax.yaxis.set_label_coords(yl_coords[0], yl_coords[1])
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.set_aspect(aspect=aspect)
    ax.set_xlim(0, len(xticks))
    # ax.set_yscale('log')
    ax.set_ylim(y_axis_lim - 0.5, 0.99)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ticks = np.arange(y_axis_lim, 1, 1)
    ax.set_yticks(ticks)
    ax.set_yticklabels([r"$10^{{{}}}$".format(tick) for tick in ticks])
    # if type(y) == str:
    #     pass
    # else:
    #     plt.yticks(y)
    ax.tick_params(axis='x', colors=color, labelsize=xl_font_scaling*ft,
                   direction='in', length = ft*0.3, width = ft * 0.1)
    ax.tick_params(axis='x', which='minor', bottom=False)
    ax.tick_params(axis='y', which='both', colors=color, labelsize=ft * 0.8,
                   direction='in', width=ft * 0.1, length=ft * 0.3)
    ax.title.set_color(color)
    plt.subplots_adjust(bottom = bottom)
    # ax.set_aspect('equal', 'box')
    return(fig, ax)


def plot_data(cell_intensities_subset, output_folder, induction_levels, probe_levels,
              signal, order, xticklabels,
              ft, color, jit, aspect, shape_inches):
    # Label the X axis field types
    xlabel = 'Probe\nExtension:\n\n Induction:'
    # xticks = list(range(1, len(induction_levels[0]) * len(probe_levels[0]) + 1))
    # Combination of both params
    num_xticks = len(induction_levels[0]) * len(probe_levels[0])
    # print(num_xticks)
    # print(spacing)
    # Create a list of xtick positions
    # xticks = np.power(np.repeat(10, num_xticks), list(np.arange(0.5, (num_xticks + 1) - 0.5, 1)))
    xticks = list(np.arange(0.5, (num_xticks + 1) - 0.5, 1))
    # ymin_list = []
    # ymax_list = []
    # for subset in order:
    #     ymin_list.append(cell_intensities_subset[subset][1].min(0))
    #     # ymax_list.append(cell_intensities_subset[subset][1].max(0))
    # ymin_all = min(ymin_list)
    # ymax_all = max(ymax_list)
    # print(list(range(1*2, (4+1)*2, 2)))
    # print(xticks)
    # Label the readout on the y axis
    ylabel = signal[1][0]
    # Initiate plot with parameters
    # fig, ax = set_plot_params(ft, x, xlabel, xticklabels, y, ylabel, color)
    fig, ax = set_plot_params(ft, xlabel, xticks, xticklabels, ylabel, color, aspect, bottom)
    # # jit = 0.15
    # # transparency = 0.5
    # position = 1
    # Iterate through data in the correct order
    for i in range(len(order)):
        # Extract the name of the data subset and the position on the x axis
        subset = order[i]
        position = xticks[i]
        #     # coeff = pd.read_csv(file)
        #     # for i in range(key_info.shape[0]):
        #     #     key = key_info.key[i]
        #     #     if key in file:
        #     #         if key == 'mch_2_mch_first':
        #     #             continue
        #     #         probes = key_info.probes[i]
        #     #         expmnt = key_info.expmnt[i]
        #     #         order = key_info.order[i]
        #     #         sub_type = key_info.sub_type[i]
        #     #         if order == 'EUB_first':
        #     #             color = 'y'
        #     #             stagger = -0.1
        #     #         elif order == 'mch_first':
        #     #             color = 'r'
        #     #             stagger = 0
        #     #         else:
        #     #             color = 'blue'
        #     #             stagger = 0.1
        #     #         coeff_1 = coeff.iloc[1:coeff.shape[0], 1].values
        #     #         jitter = np.random.normal(scale=jit, size=coeff_1.shape[0])
        #     #         x_list = np.repeat(expmnt*scale, coeff_1.shape[0]) + jitter + stagger
        #     #         scat = ax.scatter(x_list, coeff_1, alpha=transparency, c=color)
        # Get the specified data subset
        intensity_values = cell_intensities_subset[subset][1]
        # Measure the maximum and mean values
        ymax = intensity_values.max()
        ymean = np.mean(intensity_values)
        # Create a normally distributed list of random numbers
        jitter = np.random.normal(scale=jit, size=intensity_values.shape[0])
    #     # x_list = np.repeat(expmnt*scale, coeff_1.shape[0]) + jitter + stagger
        # Jitter the points in the x direction
        x_list = np.repeat(position, intensity_values.shape[0]) + jitter
        # Plot the points for the given subset
        scat = ax.scatter(x_list, np.log10(intensity_values), s=ft*0.5,
                          marker='.', edgecolors='none', alpha=transparency, c=color)
        # Plot the mean as a horizontal red bar
        scat = ax.hlines(np.log10(ymean), position -
                         jit*2, position + jit*2, colors='red', linewidth=ft * 0.15)
        fig.set_size_inches(shape_inches[0], shape_inches[1])
        # # Add the mean value above the points
        # ax.text(position, ymax, round(ymean, 4), ha='center',
        #         va='bottom', color=color, fontsize=ft * 0.8)
        # position = position + 1

    # # Print t-test result for difference between inductions for first extension param
    # # Get inputs
    # x0, x1 = xticks[0], xticks[1]
    # y0, y1 = cell_intensities_subset[order[0]][1], cell_intensities_subset[order[1]][1]
    # # Get height
    # y01, h01 = max([y0.max(), y1.max()]) + 1, 1
    # # Plot lines
    # ax.plot([x0, x0, x1, x1], [y01, y01 + h01, y01 + h01, y01], lw=ft * 0.05, c=color)
    # # Perform t-test
    # test = stats.ttest_ind(y0, y1)
    # print(test)
    # # Get p-value
    # pvalue = test[1]
    # # Create a string from t-test result
    # if pvalue < 0.000001:
    #     pvalue_text = 'p < 10^-6'
    # else:
    #     pvalue_text = 'p =' + str(pvalue)
    # # Print text on plot
    # ax.text((x0 + x1) * 0.5, y01 + h01, pvalue_text, ha='center',
    #         va='bottom', color=color, fontsize=ft * 0.8)
    #
    # # Print t-test result for difference between inductions for second extension params
    # x2, x3 = xticks[2], xticks[3]
    # y2, y3 = cell_intensities_subset[order[2]][1], cell_intensities_subset[order[3]][1]
    # y23, h23 = max([y2.max(), y3.max()]) + 1, 1
    # ax.plot([x2, x2, x3, x3], [y23, y23 + h23, y23 + h23, y23], lw=ft * 0.05, c=color)
    # test = stats.ttest_ind(y2, y3)
    # pvalue = test[1]
    # if pvalue < 0.000001:
    #     pvalue_text = 'p < 10^-6'
    # else:
    #     pvalue_text = 'p =' + str(pvalue)
    # ax.text((x2 + x3) * 0.5, y23 + h23, pvalue_text, ha='center',
    #         va='bottom', color=color, fontsize=ft * 0.8)

    # # custom_lines = [Line2D([0], [0], marker='o', color='w', label='EUB_first',
    # #                        markerfacecolor='y', mec='y', markersize=ft*0.5),
    # #                 Line2D([0], [0], marker='o', color='w', label='mch_first',
    # #                        markerfacecolor='r', mec='r', markersize=ft*0.5),
    # #                 Line2D([0], [0], marker='o', color='w', label='simultaneous',
    # #                        markerfacecolor='blue', mec='blue', markersize=ft*0.5)]
    # # plt.figtext(0.25, 0.05, '1', color='w', fontsize=ft)
    # # plt.figtext(0.75, 0.05, '9', color='w', fontsize=ft)
    # # ax.legend(loc='upper center', handles=custom_lines, fontsize=ft*0.75)
    # # plt.savefig(output_folder + '/copynumber_' + copynumber 'gel_like_plot.png',
    # #             bbox_inches='tight', transparent=True)
    # # plt.clf()
    # # plt.close()
    return(scat)


# Subset the data by specific parameters and get info for plotting
def subset_data(cell_intensities, induction_levels, probe_levels, copynumber):
    # Create an empty dictionary
    cell_intensities_subset = {}
    # Create an empty list to specify plot order
    order = []
    # Create an epmty list to specify x labels
    xticklabels = []
    # Iterate through induction levels and probe levels
    for j in range(len(probe_levels[0])):
        probe = probe_levels[0][j]
        for i in range(len(induction_levels[0])):
            induction = induction_levels[0][i]
            # Make a dictionary key
            key = "induction_" + induction + "_probe_" + probe
            # Fill out the dictionary with subset info in the first field and data in the second
            cell_intensities_subset[key] = [[signal[1][0], induction_levels[1][i], probe_levels[1][j]],
                                            cell_intensities[signal[0][0]][(cell_intensities.atc == induction) &
                                                                           (cell_intensities.probe == probe) &
                                                                           (cell_intensities.copynumber == copynumber)]]

            order.append(key)
            xticklabels.append('\n' + probe_levels[1][j] + '\n \n' + induction_levels[1][i])
            # xticklabels.append('')
            print(key)
            print(len(cell_intensities_subset[key][1]))
    print(order)
    print(xticklabels)
    return(cell_intensities_subset, order, xticklabels)


################################################################################################################################################################
# Variables
################################################################################################################################################################
# input_dir = "data/tables/cell_intensity_analyzed_with_factors.csv"
# output_folder = "figures"
# induction_levels = [["0nM", "50nM"], ["-", "+"]]
# probe_levels = [["norm", "extbranch"], ["-", "+"]]
# copynumber = "l"
# signal = [["mRNA_GFP_r.27.28.546"], ["Cell-average\nmRNA readout intensity [-]"]]
#
# ft = 20
# color = 'white'
# jit = 0.07
# transparency = 0.2
# shape_inches = [6, 6]
# aspect = 1.4

################################################################################################################################################################
# Main
################################################################################################################################################################

def main():
    parser = argparse.ArgumentParser('Blocking probe intensity analysis')
    parser.add_argument('image_dir', type = str, help='Input folder')
    # parser.add_argument('pure_spectra_folder', type=str,
    #                     help='Input folder containing csv of avgint of pure spectra')
    parser.add_argument('-o', '--out', dest = 'output_filenames', nargs = '+', type=str, help='Output filename containing plots')

    parser.add_argument('-f', '--font', dest = 'ft', type=int, default = 20, help='Font size')
    parser.add_argument('-xlf', '--xl_font_scaling', dest = 'xl_font_scaling', type=float, default = 0.9, help='Scale font size for xtick labels')
    parser.add_argument('-c', '--color', dest = 'color', type=str, default = 'black', help='plot color')
    parser.add_argument('-j', '--jitter', dest = 'jit', type=float, default = 0.07, help='Dot jitter width')
    parser.add_argument('-t', '--transparency', dest = 'transparency', type=float, default = 0.2, help='Dot transparency')
    parser.add_argument('-a', '--aspect', dest = 'aspect', type=float, default = 0.8, help='Plot aspect ratio')
    parser.add_argument('-s', '--size_inches', dest = 'size_inches', default = 6, type=float, help='plot size in inches')
    parser.add_argument('-b', '--bottom', dest = 'bottom', default = 0.2, type=float, help='space between x axis and bottom of plot.')
    parser.add_argument('-xlc', '--xl_coords', dest = 'xl_coords', nargs = 2, default = [-0.05, -0.01], type=float, help='space between x axis and bottom of plot.')
    parser.add_argument('-ylc', '--yl_coords', dest = 'yl_coords', nargs = 2, default = [-0.1, 0.5], type=float, help='space between x axis and bottom of plot.')
    parser.add_argument('-yal', '--y_axis_lim', dest = 'y_axis_lim', default = -3, type=float, help='Log10 of lower limit of y axis')

    parser.add_argument('-ilko', '--image_list_keys_ordered', dest = 'image_list_keys_ordered', nargs = '+', type=str, help='List in order the keys for the images you want to display')
    parser.add_argument('-isi', '--image_set_id', dest = 'image_set_id', type=str, help='Give the key for the image type you want to plot. If you want to plot merged lasers, pass merge.')
    parser.add_argument('-gshp', '--grid_shape', dest = 'grid_shape', nargs = 2, type=int, help='rows by columns shape for the grid display')
    parser.add_argument('-rl', '--row_labels', dest = 'row_labels', nargs = '+', type=str, help='labels to display next to the rows')
    parser.add_argument('-cl', '--column_labels', dest = 'column_labels', nargs = '+', type=str, help='labels to display above the columns')
    args = parser.parse_args()

    # Plot images in subplots grid
    # if not args.image_set_id == 'merge':
    image_filename_list = glob.glob('{}/*{}'.format(args.image_dir, args.image_set_id))
    fig = plt.figure()
    fig.set_size_inches(10,10)
    for i in range(len(args.image_list_keys_ordered)):
        key = args.image_list_keys_ordered[i]
        for f in image_filename_list:
            if key in f:
                filename = f
        ax = fig.add_subplot(args.grid_shape[0], args.grid_shape[1], i+1, frameon=False)
        image = plt.imread(filename)
        ax.imshow(image)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.xticks([])
        plt.yticks([])
        plt.gca().spines['bottom'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        # plt.axis('off')
        if i < args.grid_shape[1]:
            plt.title(args.column_labels[i], color = args.color)
        if i % args.grid_shape[1] == 0:
            plt.ylabel(args.row_labels[int(i/args.grid_shape[1])], color = args.color)
    # plt.show()
    for o in args.output_filenames:
        plt.savefig(o, dpi = 300, bbox_inches='tight', transparent=True)
    plt.close()
    # fig, ax = plt.subplots(args.grid_shape[0], args.grid_shape[1], figsize=(ft*1.5,ft*1.5))
    #
    #
    # # Get factors into a dataframe
    # factors = list(zip(*(iter(args.factors),) * 2))
    # levels = list(zip(*(iter(args.levels),) * args.nfactors * 2))
    # factors_df = pd.DataFrame(factors, columns = ['factor','factor_label'])
    # level_columns = []
    # for f in factors_df.factor.values:
    #     level_columns.extend([f, f + '_label'])
    # levels_df = pd.DataFrame(levels, columns = level_columns)
    #
    # # Set up plot parameters
    # num_xticks = levels_df.shape[0]
    # xticks = list(np.arange(0.5, (num_xticks + 1) - 0.5, 1))
    # ylabel = args.signal[1]
    # xlabel = ''
    # xticklabels = []
    # for index, level in levels_df.iterrows():
    #     xticktemp = ''
    #     for f in factors_df.factor:
    #         if not level[f + '_label'] == 'None':
    #             xticktemp += '\n' + level[f + '_label']
    #     xticklabels.append(xticktemp)
    # for f in factors_df.factor:
    #     factor_label_temp = factors_df.loc[factors_df.factor == f, 'factor_label'].values[0]
    #     if not level[f + '_label'] == 'None':
    #         xlabel += '\n' + factor_label_temp
    # fig, ax = set_plot_params(args.ft, xlabel, xticks, xticklabels, ylabel, args.color, args.aspect, args.bottom, args.xl_coords, args.yl_coords, args.xl_font_scaling, args.y_axis_lim)
    # # Plot figure
    # i = 0
    # for index, level in levels_df.iterrows():
    #     filename_factor_level_combo = ''
    #     for f in factors_df.factor.values:
    #         filename_factor_level_combo += f + '_' + level[f] + '_'
    #     for filename in args.input_filename_list:
    #         if filename_factor_level_combo in filename:
    #             intensity_values = pd.read_csv(filename)
    #             intensity_values = intensity_values[args.signal[0]]
    #             # print(intensity_values)
    #             position = xticks[i]
    #             # Measure the maximum and mean values
    #             ymax = intensity_values.max()
    #             ymean = np.mean(intensity_values)
    #             # Create a normally distributed list of random numbers
    #             jitter = np.random.normal(scale=args.jit, size=intensity_values.shape[0])
    #         #     # x_list = np.repeat(expmnt*scale, coeff_1.shape[0]) + jitter + stagger
    #             # Jitter the points in the x direction
    #             x_list = np.repeat(position, intensity_values.shape[0]) + jitter
    #             # Plot the points for the given subset
    #             scat = ax.scatter(x_list, np.log10(intensity_values), s=args.ft*0.5,
    #                               marker='.', edgecolors='none', alpha=args.transparency, c=args.color)
    #             # Plot the mean as a horizontal red bar
    #             scat = ax.hlines(np.log10(ymean), position -
    #                              args.jit*2, position + args.jit*2, colors='red', linewidth=args.ft * 0.15)
    #             i += 1
    # plt.show()
    # # Save
    # for filename in args.output_filenames:
    #     plt.savefig(filename, bbox_inches='tight', transparent=True)
    # plt.clf()
    # plt.close()

    # for i in len(factors):
    #     f = factors[i]
    #     factors_df = factors_df.append({'factor':f[0], 'factor_label':f[1]}, ignore_index = True)
    #     for l in levels:
    #         levels_df = levels_df.append({'level:f[0], 'factor_label':f[1]}, ignore_index = True)
    # # Make a column with the actual name in the filename
    # factors_df['string'] = factors_df.factor + '_' + factors_df.level + '_'
    # # Get a list of only the files specified in the factors
    # # Get factor levels for xtick labels
    # factor_filenames = []
    # xticklabels = []
    # for filename in args.input_filename_list:
    #     # i = 0
    #     bool = True
    #     # Go through all unique factors
    #     for f in factors_df.factor.unique():
    #     # while i < len(factors_df.factor.unique()):
    #         # f = factors_df.factor.unique()[i]
    #         strings = factors_df.loc[factors_df.factor == f, 'string'].copy()
    #         if factors_df.loc[factors_df.factor == f, 'level'].values[0] == 'all':
    #             pass
    #         elif not any(l in filename for l in strings):
    #             bool = False
    #         else:
    #             pass
    #         # i += 1
    #     if bool:
    #         factor_filenames.append(filename)
    #         xticktemp = ''
    #         for fact in factors_df.factor.unique():
    #             # print(fact)
    #             level = re.findall(r'(?<=' + fact + '_)[^\W_]+', filename)[0]
    #             check_level = factors_df.loc[factors_df.factor == fact, 'level'].values[0]
    #             if check_level == 'all':
    #                 level_name_temp = level
    #             else:
    #                 print(level)
    #                 level_name_temp = factors_df.loc[factors_df.level == level, 'level_label'].values[0]
    #                 print(level_name_temp)
    #             # print(level_temp)
    #             xticktemp += '\n' + level_name_temp
    #         xticklabels.append(xticktemp)
    # # print(factor_filenames, xticklabels)
    # # Set up plot parameters
    # num_xticks = len(factor_filenames)
    # xticks = list(np.arange(0.5, (num_xticks + 1) - 0.5, 1))
    # ylabel = args.signal[1]
    # xlabel = ''
    # for f in factors_df.factor.unique():
    #     factor_label_temp = factors_df.loc[factors_df.factor == f, 'factor_label'].values[0]
    #     xlabel += '\n' + factor_label_temp
    # fig, ax = set_plot_params(args.ft, xlabel, xticks, xticklabels, ylabel, args.color, args.aspect, args.bottom, args.xl_coords, args.yl_coords, args.xl_font_scaling, args.y_axis_lim)
    # # Iterate through the chosen filenames
    # for i in range(len(factor_filenames)):
    #     filename = factor_filenames[i]
    #     intensity_values = pd.read_csv(filename)
    #     intensity_values = intensity_values[args.signal[0]]
    #     print(intensity_values)
    #     position = xticks[i]
    #     # Measure the maximum and mean values
    #     ymax = intensity_values.max()
    #     ymean = np.mean(intensity_values)
    #     # Create a normally distributed list of random numbers
    #     jitter = np.random.normal(scale=args.jit, size=intensity_values.shape[0])
    # #     # x_list = np.repeat(expmnt*scale, coeff_1.shape[0]) + jitter + stagger
    #     # Jitter the points in the x direction
    #     x_list = np.repeat(position, intensity_values.shape[0]) + jitter
    #     # Plot the points for the given subset
    #     scat = ax.scatter(x_list, np.log10(intensity_values), s=args.ft*0.5,
    #                       marker='.', edgecolors='none', alpha=args.transparency, c=args.color)
    #     # Plot the mean as a horizontal red bar
    #     scat = ax.hlines(np.log10(ymean), position -
    #                      args.jit*2, position + args.jit*2, colors='red', linewidth=args.ft * 0.15)
    #     print(np.log10(intensity_values))
    # plt.show()




# # Read in data file
# cell_intensities = pd.read_csv(input_filename)
# print(cell_intensities.shape)
#
# # cell_intensities_subset = {}
# # for i in range(len(induction_levels[0])):
# #     induction = induction_levels[0][i]
# #     for j in range(len(probe_levels[0])):
# #         probe = probe_levels[0][i]
# #         cell_intensities_subset["induction_" + induction + "_probe_" + probe] = \
# #             [[signal[1][0], induction_levels[1][i], probe_levels[1][i]],
# #              cell_intensities[signal[0][0]][(cell_intensities.atc == induction) &
# #                                             (cell_intensities.probe == probe) &
# #                                             (cell_intensities.copynumber == copynumber)]]
#
# # Get Subsets and info for plotting
# cell_intensities_subset, order, xticklabels = subset_data(
#     cell_intensities, induction_levels, probe_levels, copynumber)
# # print(len(cell_intensities_subset["induction_0nM_probe_norm"][1]))
# # print(cell_intensities_subset["induction_0nM_probe_norm"][0])
#
# # Plot the data
# plot_data(cell_intensities_subset, output_folder,
#           induction_levels, probe_levels, signal, order, xticklabels, ft, color, jit, aspect, shape_inches)
#
# # Set up a naming pattern to record the factors used in this plot
# name_induction_levels = 'induction_levels_'
# name_probe_levels = 'probe_levels_'
# for i in induction_levels[0]:
#     name_induction_levels = name_induction_levels + i + '_'
#     print(i)
#     print(name_induction_levels)
# for j in probe_levels[0]:
#     name_probe_levels = name_probe_levels + j + '_'
#     print(j)
#     print(name_probe_levels)
#
# # Save as a pdf and png
# plt.savefig(output_folder + '/copynumber_' + copynumber + '_' + name_induction_levels + name_probe_levels + 'gel_like_plot.png',
#             bbox_inches='tight', transparent=True)
# plt.savefig(output_folder + '/copynumber_' + copynumber + '_' + name_induction_levels + name_probe_levels + 'gel_like_plot.pdf',
#             bbox_inches='tight', transparent=True)
# plt.clf()
# plt.close()

if __name__ == '__main__':
    main()
