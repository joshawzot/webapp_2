# Standard library imports
from datetime import datetime
from io import BytesIO
import base64
import re

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import scipy.stats as sp_stats  # Alias for scipy.stats
from scipy.stats import gamma
from scipy.integrate import quad
from fitter import Fitter

# Local application imports
from db_operations import create_connection, close_connection

def calculate_statistics(data):
    """Calculate statistics for a given set of data and round to two decimal places."""
    mean = round(np.mean(data), 2)
    std_dev = round(np.std(data), 2)
    outlier_percentage = round(np.sum(np.abs(data - mean) > 2.698 * std_dev) / len(data) * 100, 2)
    return mean, std_dev, outlier_percentage

def plot_boxplot(data, table_names, figsize=(15, 10)):
    plt.figure(figsize=figsize) 
    # Boxplot setup

    xticks = []
    xticklabels = table_names  # Adjusted to your groups' names
    
    #print("data:", data)
    # Iterate over each group to plot
    for i, group in enumerate(data):
        # The first position of each group for the x-tick
        start_position = i * len(group) + 1
        xticks.append(start_position)

        for j, data in enumerate(group):
            position = i * len(group) + j + 1
            plt.boxplot(data, positions=[position], widths=0.6)

            '''# Calculate and annotate statistics
            mean, std_dev, outlier_percentage = calculate_statistics(data)
            offset = 10  # Distance above the max value for annotation
            plt.text(position, max(data) + offset, f'Avg: {mean:.2f}\nSD: {std_dev:.2f}\nOutliers: {outlier_percentage}%',
                     horizontalalignment='center', verticalalignment='top', fontsize=11)'''

    # Ensure xticks and xticklabels are matched
    if len(xticks) == len(xticklabels):
        plt.xticks(xticks, xticklabels, rotation=45, fontsize=12)
    else:
        print("Error: Mismatch in the number of xticks and xticklabels.")

    plt.yticks(fontsize=12)
    #plt.ylabel('Conductance (u)', fontsize=12)
    plt.grid(True)

    # Save the boxplot figure to a buffer
    buf_boxplot = BytesIO()
    plt.savefig(buf_boxplot, format='png', bbox_inches='tight')
    buf_boxplot.seek(0)
    plot_data_boxplot = base64.b64encode(buf_boxplot.getvalue()).decode('utf-8')
    buf_boxplot.close()

    # Clear the figure to start fresh for another plot
    plt.clf()

    return plot_data_boxplot

def plot_histogram(data, table_names, colors, figsize=(15, 10)):
    plt.figure(figsize=figsize)
    
    # Calculate global min and max across all groups and subgroups for setting the histogram range
    global_min = min([min(subgroup) for group in data for subgroup in group])
    global_max = max([max(subgroup) for group in data for subgroup in group])

    # Create a list of bin edges with an increment of 1, using global_min and global_max
    bin_edges = np.arange(global_min, global_max + 1, 1)  # +1 to include the last value

    # Track the filenames that have been added to the legend
    added_to_legend = set()

    # Iterate over each group to plot
    for i, group in enumerate(data):
        for j, subgroup in enumerate(group):
            # Only add the label the first time a filename is encountered
            label = f'{table_names[i]}' if table_names[i] not in added_to_legend else None
            if label:
                added_to_legend.add(table_names[i])

            # Generate histogram for each subgroup with the specified bin edges
            plt.hist(subgroup, bins=bin_edges, color=colors[i], alpha=0.75, label=label, log=True)
    
    #plt.xlabel('Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    #plt.legend()
    #plt.legend(loc='upper right')

    # Save the histogram figure to a buffer
    buf_histogram = BytesIO()
    plt.savefig(buf_histogram, format='png', bbox_inches='tight')
    buf_histogram.seek(0)
    plot_data_histogram = base64.b64encode(buf_histogram.getvalue()).decode('utf-8')
    buf_histogram.close()

    # Clear the figure to start fresh for another plot
    plt.clf()

    return plot_data_histogram

'''
I want to also get the BER between different states for each table_names, for example for a specific table_name,
if it has 4 states of data, state0, state1, state2, state3, there will be 3 BER, one between state0 and state1, between state1 and state2, between state2 and state3.
To calculate the BER,
reverse the y-axis (for example 1 becomes -1, -2 becomes 2) of state0 and record the absolute value of y-axis that the transformed cdf plot of state0 and the transformed cdf plot of state1 intersects.
reverse the y-axis (for example 1 becomes -1, -2 becomes 2) of state1 and record the absolute value of y-axis that the transformed cdf plot of state1 and the transformed cdf plot of state2 intersects.
reverse the y-axis (for example 1 becomes -1, -2 becomes 2) of state2 and record the absolute value of y-axis that the transformed cdf plot of state2 and the transformed cdf plot of state3 intersects.
the absolute value of y-axis of intersects are the BER.
'''

def sigma_to_ppm(sigma):
    # Calculate the area in the tail beyond the sigma value on one side of the distribution
    tail_probability = sp_stats.norm.sf(sigma)
    # Convert this probability to parts per million
    ppm = tail_probability * 1_000_000
    return ppm

from scipy.stats import norm 
def cdf_to_ppm(cdf): 
    tail_probability = cdf
    ppm = tail_probability * 1_000_000 
    return ppm

from scipy.stats import norm 
def sigma_to_cdf(sigma): 
    # Calculate the CDF for the given sigma value 
    cdf_value = norm.cdf(sigma) 
    return cdf_value

from scipy.interpolate import interp1d
def plot_transformed_cdf_2(data, table_names, selected_groups, colors, figsize=(15, 10)):
    # First plot: Transformed CDF
    plt.figure(figsize=figsize)
    added_to_legend = set()
    ber_results = []
    transformed_data_groups = []
    global_x_min = float('inf')
    global_x_max = float('-inf')

    # Create separate figures for sigma and CDF plots with specified size
    fig_sigma, ax_sigma = plt.subplots(figsize=figsize)
    fig_cdf, ax_cdf = plt.subplots(figsize=figsize)

    for i, group in enumerate(data):
        transformed_data = []

        for j, subgroup in enumerate(group):
            state_index = selected_groups[j]
            table_name = table_names[i]
            label = table_name if table_name not in added_to_legend else None
            if label:
                added_to_legend.add(label)

            sorted_data = np.sort(subgroup)
            global_x_min = min(global_x_min, sorted_data[0])  # Update global x-axis minimum
            global_x_max = max(global_x_max, sorted_data[-1])  # Update global x-axis maximum

            cdf_values = (np.arange(1, len(sorted_data) + 1) - 0.5) / len(sorted_data)
            sigma_values = sp_stats.norm.ppf(cdf_values)

            # Plot sigma values
            ax_sigma.plot(sorted_data, sigma_values, linestyle='-', linewidth=1, color=colors[i], label=label)
            ax_sigma.scatter(sorted_data, sigma_values, s=10, color=colors[i])

            # Plot CDF values
            ax_cdf.plot(sorted_data, cdf_values, linestyle='-', linewidth=1, color=colors[i], label=label)
            ax_cdf.scatter(sorted_data, cdf_values, s=10, color=colors[i])

            transformed_data.append((sorted_data, sigma_values))

        transformed_data_groups.append(transformed_data)

    # Sigma plot settings
    #ax_sigma.set_title('Sigma Plot of Transformed Data by Groups')
    #ax_sigma.set_yscale('log')
    #ax_sigma.legend()
    ax_sigma.grid(True)

    # CDF plot settings
    #ax_cdf.set_title('CDF Plot of Transformed Data by Groups')
    ax_cdf.set_yscale('log')
    #ax_cdf.legend()
    ax_cdf.grid(True)
    # Set fixed x-axis limits for CDF plot
    #ax_cdf.set_xlim(0, 200)  # Setting x-axis minimum to 0 and maximum to 200
    
    # Save sigma plot
    buf_sigma = BytesIO()
    fig_sigma.savefig(buf_sigma, format='png', bbox_inches='tight')
    buf_sigma.seek(0)
    plot_data_sigma = base64.b64encode(buf_sigma.getvalue()).decode('utf-8')
    buf_sigma.close()

    # Save CDF plot
    buf_cdf = BytesIO()
    fig_cdf.savefig(buf_cdf, format='png', bbox_inches='tight')
    buf_cdf.seek(0)
    plot_data_cdf = base64.b64encode(buf_cdf.getvalue()).decode('utf-8')
    buf_cdf.close()

    # Clear the figures after saving
    plt.close(fig_sigma)
    plt.close(fig_cdf)

    plt.figure(figsize=figsize)
    plt.xlim(global_x_min, global_x_max)  # Set the x-axis to match the original plot's range

    intersections = []
    horizontal_line_y_value = []

    for i, transformed_data in enumerate(transformed_data_groups):
        print("i:", i)

        for k in range(len(transformed_data) - 1):
            x1, y1 = transformed_data[k]
            x2, y2 = transformed_data[k + 1]

            y1 = -y1  # Reverse the y-axis for the first of the two states being compared

            start_state = selected_groups[k]
            end_state = selected_groups[k + 1]

            common_x_min_all = min(min(x1), min(x2))
            common_x_max_all = max(max(x1), max(x2))
            common_x_all = np.linspace(common_x_min_all, common_x_max_all, num=5000)

            # Remove duplicates and interpolate
            unique_x1, unique_indices_x1 = np.unique(x1, return_index=True)
            unique_y1 = y1[unique_indices_x1]
            unique_x2, unique_indices_x2 = np.unique(x2, return_index=True)
            unique_y2 = y2[unique_indices_x2]

            interp_common_x_1 = interp1d(unique_x1, unique_y1, fill_value="extrapolate")(common_x_all)
            interp_common_x_2 = interp1d(unique_x2, unique_y2, fill_value="extrapolate")(common_x_all)

            # Convert sigma to CDF values
            #cdf_value_1 = norm.cdf(interp_common_x_1)
            #cdf_value_2 = norm.cdf(interp_common_x_2)

            # Don't Convert sigma to CDF values, keep sigma values
            cdf_value_1 = interp_common_x_1
            cdf_value_2 = interp_common_x_2

            # Check if both cdf_value_1 and cdf_value_2 are not all NaN before plotting and finding intersections
            if not (np.isnan(cdf_value_1).all() or np.isnan(cdf_value_2).all()):
                plt.plot(common_x_all, cdf_value_1, linestyle='-', color=colors[i], alpha=0.7)
                plt.plot(common_x_all, cdf_value_2, linestyle='-', color=colors[i], alpha=0.7)

                # Find and mark intersection only if both arrays have valid data
                idx_closest = np.argmin(np.abs(cdf_value_1 - cdf_value_2))
                intersection_x = common_x_all[idx_closest]
                intersection_y = cdf_value_1[idx_closest]
                plt.scatter(intersection_x, intersection_y, color='red', s=50, zorder=5)
                intersections.append((intersection_x, intersection_y))

                ber = np.abs(cdf_value_1[idx_closest])
                #ppm_ber = cdf_to_ppm(ber)  # Assuming sigma_to_ppm is defined elsewhere
                ppm_ber = sigma_to_ppm(ber)
                #ppm_ber = ber

                # Draw horizontal lines if x-differences are about 2 units apart
                target_x_diff = 2
                tolerance = 0.1
                line_drawn = False

                for idx in range(len(common_x_all) - 1):
                    for jdx in range(idx + 1, len(common_x_all)):
                        x_diff = common_x_all[jdx] - common_x_all[idx]
                        if abs(x_diff - target_x_diff) < tolerance:
                            if cdf_value_2[jdx] > cdf_value_1[idx]:  # Check divergence
                                plt.hlines(y=cdf_value_2[jdx], xmin=common_x_all[idx], xmax=common_x_all[jdx], color='green', linestyles='dotted')
                                horizontal_line_y_value = cdf_value_2[jdx]
                                #ppm = cdf_to_ppm(abs(horizontal_line_y_value))
                                ppm = sigma_to_ppm(abs(horizontal_line_y_value))
                                print(f"Horizontal line drawn from x={common_x_all[idx]} to x={common_x_all[jdx]} at y={cdf_value_2[jdx]}")
                                print("ppm:", ppm)
                                line_drawn = True
                                break
                    if line_drawn:
                        break

                if not line_drawn:
                    print("No suitable points found to draw a horizontal line.")

                ber_results.append(('random', f'state{start_state} to state{end_state}', ber, ppm_ber, ppm, round(abs(horizontal_line_y_value), 4)))
        
    #plt.ylabel('CDF Value', fontsize=12)
    #plt.title('CDF Curves for BER Calculation')
    plt.grid(True)
    #plt.yscale('log')
    #plt.ylim(bottom=1e-6, top=100)  # Set the lower limit of y-axis to 10^-6
    plt.ylim(bottom=-8, top=8)  # Set the lower limit of y-axis to 10^-6

    # Save the figure to a buffer and encode as base64 for embedding or saving
    buf_interpolated_cdf = BytesIO()
    plt.savefig(buf_interpolated_cdf, format='png', bbox_inches='tight')
    buf_interpolated_cdf.seek(0)
    plot_data_interpolated_cdf = base64.b64encode(buf_interpolated_cdf.getvalue()).decode('utf-8')
    buf_interpolated_cdf.close()
    plt.clf()

    print("ber_results:", ber_results)
    #print("horizontal_line_y_value:", horizontal_line_y_value)
    return plot_data_sigma, plot_data_cdf, plot_data_interpolated_cdf, ber_results

from db_operations import create_connection, fetch_data, close_connection, create_db_engine, create_db, get_all_databases, connect_to_db, fetch_tables, rename_database, move_tables, copy_tables, copy_all_tables, copy_tables_2, move_tables_2
def get_group_data_new(table_name, selected_groups, database_name, sub_array_size):
    connection = create_connection(database_name)
    query = f"SELECT * FROM `{table_name}`"
    cursor = connection.cursor()
    cursor.execute(query)
    data = cursor.fetchall()

    # Convert fetched data to a NumPy array for easier manipulation
    data_np = np.array(data)

    data_np[data_np == 0] = 0.001

    if np.mean(data_np) < 1:
        data_np = data_np * 1e6

    groups = []
    groups_stats = []  # List to store statistics for each group

    rows_per_group, cols_per_group = sub_array_size
    total_rows, total_cols = data_np.shape

    num_row_groups = total_rows // rows_per_group
    num_col_groups = total_cols // cols_per_group
    num_of_groups = num_col_groups * num_row_groups
    partial_rows = total_rows % rows_per_group  # Check if there's a partial row group
    partial_cols = total_cols % cols_per_group  # Check if there's a partial column group

    group_idx = 0  # Initialize group index
    real_selected_groups = []

    for i in range(num_row_groups + (1 if partial_rows > 0 else 0)):
        for j in range(num_col_groups + (1 if partial_cols > 0 else 0)):
            start_row = i * rows_per_group
            end_row = (i + 1) * rows_per_group if i < num_row_groups else total_rows

            start_col = j * cols_per_group
            end_col = (j + 1) * cols_per_group if j < num_col_groups else total_cols

            # Check if this group is selected
            if group_idx in selected_groups:
                real_selected_groups.append(group_idx)

                try:
                    group = data_np[start_row:end_row, start_col:end_col]
                    flattened_group = group.flatten()

                    # Filter out negative values
                    positive_flattened_group = flattened_group[flattened_group >= 0]

                    groups.append(positive_flattened_group)

                    # Calculate statistics for the positive values
                    if len(positive_flattened_group) > 0:  # Ensure there are positive values to analyze
                        average = round(np.mean(positive_flattened_group), 2)
                        std_dev = round(np.std(positive_flattened_group), 2)
                        outlier_percentage = round(np.sum(np.abs(positive_flattened_group - average) > 2.698 * std_dev) / len(positive_flattened_group) * 100, 2)
                        groups_stats.append((table_name, group_idx, average, std_dev, outlier_percentage))
                    else:
                        print(f"State {group_idx} has no positive values for analysis.")
                except IndexError as e:
                    print(f"Error accessing data slice: {e}")

            group_idx += 1  # Increment group index after each inner loop

    close_connection()

    def transform_list_by_order(lst):
        sorted_indices = sorted(range(len(lst)), key=lambda x: lst[x])
        transformation = [0] * len(lst)
        for rank, index in enumerate(sorted_indices):
            transformation[index] = rank
        return transformation

    # Sort the groups, groups_stats, and real_selected_groups based on the average value of the group
    groups_stats.sort(key=lambda x: x[2])  # Sort by average value
    sorted_indices = [i[1] for i in groups_stats]  #Get the sorted indices
    print("sorted_indices:", sorted_indices)
    sorted_indices = transform_list_by_order(sorted_indices)
    groups = [groups[i] for i in sorted_indices]
    #real_selected_groups = [real_selected_groups[i] for i in sorted_indices]
    #print('real_selected_groups:', real_selected_groups)

    return groups, groups_stats, real_selected_groups

import itertools
def plot_histogram_with_fitting(data, table_names, colors, figsize=(15, 10)):
    encoded_plots_density = []  # Density plots data
    encoded_plots_counts = []  # Counts plots data
    encoded_plots_percentage = []  # Percentage plots data
    fitting_params = {}  # Dictionary to store fitting parameters for each table

    for table_index, table_data in enumerate(data):
        # Setup for three plot types: density, counts, and percentage
        fig_density, ax_density = plt.subplots(figsize=figsize)
        fig_counts, ax_counts = plt.subplots(figsize=figsize)
        fig_percentage, ax_percentage = plt.subplots(figsize=figsize)
        ax_density.set_title(f'Density and Best Fit for {table_names[table_index]}')
        ax_counts.set_title(f'Counts with Log Scale for {table_names[table_index]}')
        ax_percentage.set_title(f'Percentage for {table_names[table_index]}')

        table_fitting_params = {}

        for state_index, subgroup in enumerate(table_data):
            color = colors[state_index % len(colors)]

            # Density plot
            n, bins, patches = ax_density.hist(subgroup, bins=50, color=color, alpha=0.6, density=True, label=f'State {state_index + 1} (Density)')

            # Counts plot
            counts, _, _ = ax_counts.hist(subgroup, bins=bins, color=color, alpha=0.6, log=True, label=f'State {state_index + 1} (Counts)')

            # Percentage plot
            total = sum(counts)
            percentage_weights = np.ones_like(subgroup) / total
            ax_percentage.hist(subgroup, bins=bins, weights=percentage_weights * 100, color=color, alpha=0.6, label=f'State {state_index + 1} (Percentage)')
            ax_percentage.set_yscale('log')

            # Fitting process for each subgroup
            f = Fitter(subgroup, distributions=['norm', 'expon', 'gamma', 'lognorm', 'beta'], timeout=30)
            f.fit()
            best_fit_name = list(f.get_best(method='sumsquare_error').keys())[0]
            best_fit_params = f.fitted_param_[best_fit_name]  # Corrected to use fitted_param_ with underscore

            table_fitting_params[state_index] = {'distribution': best_fit_name, 'parameters': best_fit_params}

            dist = getattr(stats, best_fit_name)
            lower_bound, upper_bound = dist.ppf([0.0000001, 0.9999999], *best_fit_params[:-2], loc=best_fit_params[-2], scale=best_fit_params[-1])
            x_range = np.linspace(lower_bound, upper_bound, 1000)
            pdf_values = dist.pdf(x_range, *best_fit_params[:-2], loc=best_fit_params[-2], scale=best_fit_params[-1])

            # Plotting best fit for all plots
            ax_density.plot(x_range, pdf_values, linewidth=2, color=color, label=f'Best fit - State {state_index + 1}: {best_fit_name}')
            count_fit_values = pdf_values * np.diff(bins)[0] * total
            ax_counts.plot(x_range, count_fit_values, linewidth=2, color=color, label=f'Best fit - State {state_index + 1}: {best_fit_name}')
            percentage_fit_values = pdf_values * np.diff(bins)[0] * 100
            ax_percentage.plot(x_range, percentage_fit_values, linewidth=2, color=color, label=f'Best fit - State {state_index + 1}: {best_fit_name}')

        # Process and encode plots for all three types
        for fig, buf_list in [(fig_density, encoded_plots_density), (fig_counts, encoded_plots_counts), (fig_percentage, encoded_plots_percentage)]:
            buf = BytesIO()
            plt.tight_layout()
            fig.savefig(buf, format='png', bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
            buf_list.append(plot_data)
            buf.close()

        fitting_params[table_names[table_index]] = table_fitting_params

    # Returning encoded plot data and fitting parameters
    print("fitting_params:", fitting_params)
    return encoded_plots_density, encoded_plots_percentage, fitting_params

def get_column_widths(table_data):
    """
    Calculate column widths based on the content length of each column, aiming to
    ensure all content, especially in the first row, fits well.
    Additionally, print the width of each cell and the maximum width for each column.
    """

    max_widths = []
    for column in zip(*table_data):
        cell_widths = [len(str(item)) for item in column]  # Calculate width for each cell in the column
        max_width = max(cell_widths)
        max_widths.append(max_width)
        #print(f"Cell widths in column: {cell_widths} -> Max width: {max_width}")
        
    #print("Max widths for all columns:", max_widths)
    
    def adjusted_length(item):
        # Count special characters and capital letters as 2
        return sum(2 if (not char.islower()) else 1 for char in str(item))
    
    max_widths = [max(adjusted_length(item) for item in column) for column in zip(*table_data)]
    #print(max_widths)
    
    # Normalize widths by the length of the longest cell to get relative sizes.
    max_total_width = sum(max_widths)
    column_widths = [width / max_total_width for width in max_widths]
    
    return column_widths

def plot_average_values_table(avg_values, table_names, selected_groups, figsize=(15, 10)):
    fig, ax = plt.subplots(figsize=figsize)
    #ax.axis('tight')
    ax.axis('off')

    # Extend header to include 'Row Avg' and 'Row Std Dev'
    header = ["State"] + [f"{table_name}" for table_name in table_names] + ["Row Avg", "Row Std Dev"]
    table_data = [header]

    # Store column-wise data for calculating column averages and std dev later
    column_data = [[] for _ in table_names]

    for i, group in enumerate(selected_groups):
        row = [f"State {group}"]
        row_data = []  # For calculating row statistics
        
        for j, table_avg in enumerate(avg_values):
            avg = table_avg[i]  # Access the average for this group in the current table
            row += [f"{avg:.2f}"]
            row_data.append(avg)
            column_data[j].append(avg)

        # Calculate and append row average and std dev
        row_avg = np.mean(row_data)
        row_std = np.std(row_data)
        row += [f"{row_avg:.2f}", f"{row_std:.2f}"]
        
        table_data.append(row)

    # Calculate and append column averages and std dev
    col_avgs = [np.mean(col) for col in column_data]
    col_stds = [np.std(col) for col in column_data]
    table_data.append(["Col Avg"] + [f"{avg:.2f}" for avg in col_avgs] + ["-", "-"])  # Append "-" for the last two columns
    table_data.append(["Col Std Dev"] + [f"{std:.2f}" for std in col_stds] + ["-", "-"])

    column_widths = get_column_widths(table_data)

    table = ax.table(cellText=table_data, loc='center', colWidths=column_widths, cellLoc='center')
    #table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)

    plt.title('Average table')
    #plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    encoded_image = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return encoded_image

def print_average_values_table(avg_values, table_names, selected_groups):
    # Extend header to include 'Row Avg' and 'Row Std Dev'
    header = ["State"] + [f"{table_name}" for table_name in table_names] + ["Row Avg", "Row Std Dev"]
    print('\t'.join(header))

    # Store column-wise data for calculating column averages and std dev later
    column_data = [[] for _ in table_names]

    for i, group in enumerate(selected_groups):
        row = [f"State {group}"]
        row_data = []  # For calculating row statistics
        
        for j, table_avg in enumerate(avg_values):
            avg = table_avg[i]  # Access the average for this group in the current table
            row.append(f"{avg:.2f}")
            row_data.append(avg)
            column_data[j].append(avg)

        # Calculate and append row average and std dev
        row_avg = np.mean(row_data)
        row_std = np.std(row_data)
        row.extend([f"{row_avg:.2f}", f"{row_std:.2f}"])
        
        print('\t'.join(row))

    # Calculate and append column averages and std dev
    col_avgs = [np.mean(col) for col in column_data]
    col_stds = [np.std(col) for col in column_data]
    footer_avg = ["Col Avg"] + [f"{avg:.2f}" for avg in col_avgs] + ["-", "-"]
    footer_std = ["Col Std Dev"] + [f"{std:.2f}" for std in col_stds] + ["-", "-"]

    print('\t'.join(footer_avg))
    print('\t'.join(footer_std))

def plot_std_values_table(std_values, table_names, selected_groups, figsize=(15, 10)):
    fig, ax = plt.subplots(figsize=figsize)
    #ax.axis('tight')
    ax.axis('off')

    header = ["State"] + [f"{table_name}" for table_name in table_names] + ["Row Avg", "Row Std Dev"]
    table_data = [header]

    column_data = [[] for _ in table_names]

    for i, group in enumerate(selected_groups):
        row = [f"State {group}"]
        row_data = []
        
        for j, table_std in enumerate(std_values):
            std = table_std[i]
            row += [f"{std:.2f}"]
            row_data.append(std)
            column_data[j].append(std)

        # Calculate and append row average and std dev
        row_avg = np.mean(row_data)
        row_std = np.std(row_data)
        row += [f"{row_avg:.2f}", f"{row_std:.2f}"]
        
        table_data.append(row)

    # Calculate and append column averages and std dev
    col_avgs = [np.mean(col) for col in column_data]
    col_stds = [np.std(col) for col in column_data]
    table_data.append(["Col Avg"] + [f"{avg:.2f}" for avg in col_avgs] + ["-", "-"])
    table_data.append(["Col Std Dev"] + [f"{std:.2f}" for std in col_stds] + ["-", "-"])

    column_widths = get_column_widths(table_data)

    table = ax.table(cellText=table_data, loc='center', colWidths=column_widths, cellLoc='center')
    #table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    plt.title('Sigma table')
    #plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    encoded_image = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return encoded_image

def plot_colormap(data, title, figsize=(30, 15)):
    # Set minimum value to 0 and maximum value to the largest value in the data
    g_range = [0, np.max(data)]
    
    fig, ax = plt.subplots(figsize=figsize)
    # Apply the g_range for color scaling
    cax = ax.imshow(data, cmap=plt.cm.viridis, origin="lower", vmin=g_range[0], vmax=g_range[1])

    fig.colorbar(cax)
    
    #ax.set_title(title)
    ax.set_title(title, fontsize=12)
    
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    encoded_image = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return encoded_image

def get_full_table_data(table_name, database_name):
    connection = create_connection(database_name)
    query = f"SELECT * FROM `{table_name}`"
    cursor = connection.cursor()
    cursor.execute(query)
    data = cursor.fetchall()
    close_connection()  # Make sure to pass the connection object to properly close it
    
    # Assuming the data is structured as a list of tuples, where each tuple represents a row in the table
    data_matrix = np.array(data)

    data_matrix[data_matrix == 0] = 0.001   #0 is not working, why?
    
    # Get the size (shape) of the matrix
    data_matrix_size = data_matrix.shape

    return data_matrix, data_matrix_size

def calculate_overlap_fit(fit_params, selected_pairs):
    overlap_ppm = {}

    # Helper function to get the minimum PDF value between two distributions at x
    def min_pdf(x, dist1, params1, dist2, params2):
        return min(dist1.pdf(x, *params1), dist2.pdf(x, *params2))
    
    for pair in selected_pairs:
        key1, key2 = pair
        if key1 in fit_params and key2 in fit_params:
            print("key1:", key1)
            print("key2:", key2)
            dist_name1, params1 = fit_params[key1]['distribution'], fit_params[key1]['parameters']
            dist_name2, params2 = fit_params[key2]['distribution'], fit_params[key2]['parameters']

            # Get the distribution objects
            dist1 = getattr(sp_stats, dist_name1)
            dist2 = getattr(sp_stats, dist_name2)

            # Calculate the integration bounds
            lower_bound = max(dist1.ppf(0.0000001, *params1), dist2.ppf(0.0000001, *params2))
            upper_bound = min(dist1.ppf(0.9999999, *params1), dist2.ppf(0.9999999, *params2))

            # Calculate overlap using quad and min_pdf function
            if upper_bound > lower_bound:
                overlap, _ = quad(min_pdf, lower_bound, upper_bound, args=(dist1, params1, dist2, params2))
                #overlap_ppm[pair] = max(overlap * 1e6, 0)  # Convert to parts per million (ppm)
                overlap_ppm[pair] = max(overlap * 100, 0)  # Convert to percentage
            else:
                overlap_ppm[pair] = 0
        else:
            overlap_ppm[pair] = 0
    print("overlap_ppm:", overlap_ppm)
    return overlap_ppm

def plot_curve_overlap_table(overlap_ppm_by_table, selected_groups_by_table, figsize=(12, 8)):
    print("Generating combined overlap plot...")
    encoded_images = []
    combined_table_data = {}
    tables_order = []

    # Debug: Check input data
    print("overlap_ppm_by_table:", overlap_ppm_by_table)
    print("selected_groups_by_table:", selected_groups_by_table)

    # Accumulate all overlap data
    for table_name, overlap_ppm in overlap_ppm_by_table.items():
        selected_groups = selected_groups_by_table[table_name]
        tables_order.append(table_name)

        for i in range(len(selected_groups) - 1):
            state_a = selected_groups[i]
            state_b = selected_groups[i + 1]
            pair_key = (state_a, state_b)
            state_pair = f"State {state_a} & State {state_b}"
            ppm_key = next((k for k in overlap_ppm if k == (i, i + 1)), None)
            overlap_area_ppm = overlap_ppm[ppm_key] if ppm_key in overlap_ppm else 0
            
            if state_pair not in combined_table_data:
                combined_table_data[state_pair] = [0] * len(overlap_ppm_by_table)
            combined_table_data[state_pair][tables_order.index(table_name)] = overlap_area_ppm

    header = ["State Pair"] + tables_order
    table_data = [header] + [[pair] + [f"{val:.4f}%" for val in vals] for pair, vals in combined_table_data.items()]

    # Calculate the average for each column
    averages = np.mean([[float(val[:-1]) for val in row[1:]] for row in table_data[1:]], axis=0)
    averages_row = ["Average"] + [f"{avg:.4f}%" for avg in averages]
    table_data.append(averages_row)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=table_data, loc='center', colWidths=get_column_widths(table_data), cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)
    ax.set_title('Combined Overlap for All Tables')

    #plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    encoded_image = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    print("Plot generated.")
    return encoded_image

def plot_overlap_statistics(overlap_ppm_by_table, table_names):
    """
    Generates a plot of the trend of maximum, mean, and minimum overlap values across tables
    and returns the plot as a base64-encoded string.

    :param overlap_ppm_by_table: Dictionary with table names as keys and dictionaries of overlap values as values.
    :param table_names: List of table names to consider for plotting.
    :return: Base64-encoded string of the plot image.
    """
    overlap_stats = {"max": [], "mean": [], "min": []}  # To store statistics

    for table_name in table_names:
        overlaps = list(overlap_ppm_by_table[table_name].values())
        if overlaps:  # Ensure there are overlaps to calculate stats from
            max_val = max(overlaps)
            mean_val = np.mean(overlaps)
            min_val = min(overlaps)
        else:  # Default values if no overlaps
            max_val = mean_val = min_val = 0
        
        overlap_stats["max"].append(max_val)
        overlap_stats["mean"].append(mean_val)
        overlap_stats["min"].append(min_val)

    plt.figure(figsize=(10, 6))
    x_ticks = range(len(table_names))
    plt.plot(x_ticks, overlap_stats["max"], marker='o', linestyle='-', label='Max Overlap')
    plt.plot(x_ticks, overlap_stats["mean"], marker='s', linestyle='--', label='Mean Overlap')
    plt.plot(x_ticks, overlap_stats["min"], marker='^', linestyle=':', label='Min Overlap')
    plt.xticks(x_ticks, table_names, rotation=45)
    plt.xlabel('Table Name')
    #plt.ylabel('Overlap PPM Values')
    plt.ylabel('Overlap percentage Values')
    #plt.title('Overlap PPM Statistics Across Tables')
    plt.legend()
    plt.tight_layout()

    # Save the plot to a bytes buffer and then encode to base64
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()  # Close the plot to free memory
    buf.seek(0)  # Seek to the start of the bytes buffer
    encoded_image = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    return encoded_image
    
def plot_ber_tables(ber_results, table_names):
    sigma_headers = ["State/Transition"] + [name for name in table_names] + ["Row Avg"]
    ppm_headers = ["State/Transition"] + [name for name in table_names] + ["Row Avg"]
    uS_headers = ["State/Transition"] + [name for name in table_names] + ["Row Avg"]
    additional_data_headers = ["State/Transition"] + [name for name in table_names] + ["Row Avg"]

    sigma_data = [sigma_headers]
    ppm_data = [ppm_headers]
    uS_data = [uS_headers]
    additional_data = [additional_data_headers]

    grouped_data = {}
    for entry in ber_results:
        key = entry[1]
        if key not in grouped_data:
            grouped_data[key] = []
        grouped_data[key].append((entry[2], entry[3], entry[4], entry[5]))

    for key, values in grouped_data.items():
        sigma_row = [key]
        ppm_row = [key]
        uS_row = [key]
        additional_row = [key]
        for sigma, ppm, uS, additional in values:
            sigma_row.append(f"{sigma:.4f}")
            ppm_row.append(f"{ppm:.0f}")
            uS_row.append(f"{uS:.0f}")
            additional_row.append(str(additional))
        sigma_row.append(f"{np.mean([float(val) for val in sigma_row[1:] if val != key]):.4f}")
        ppm_row.append(f"{np.mean([float(val) for val in ppm_row[1:] if val != key]):.0f}")
        uS_row.append(f"{np.mean([float(val) for val in uS_row[1:] if val != key]):.0f}")
        additional_row.append(f"{np.mean([float(val) for val in additional_row[1:] if val != key]):.4f}")
        sigma_data.append(sigma_row)
        ppm_data.append(ppm_row)
        uS_data.append(uS_row)
        additional_data.append(additional_row)

    sigma_col_avg = ["Col Avg"]
    ppm_col_avg = ["Col Avg"]
    uS_col_avg = ["Col Avg"]
    additional_col_avg = ["Col Avg"]
    for col in range(1, len(sigma_data[0])):
        sigma_col = [float(row[col]) for row in sigma_data[1:] if row[col] != "Col Avg"]
        ppm_col = [float(row[col]) for row in ppm_data[1:] if row[col] != "Col Avg"]
        uS_col = [float(row[col]) for row in uS_data[1:] if row[col] != "Col Avg"]
        additional_col = [float(row[col]) for row in additional_data[1:] if row[col] != "Col Avg"]
        sigma_col_avg.append(f"{np.mean(sigma_col):.4f}")
        ppm_col_avg.append(f"{np.mean(ppm_col):.0f}")
        uS_col_avg.append(f"{np.mean(uS_col):.0f}")
        additional_col_avg.append(f"{np.mean(additional_col):.4f}")
    sigma_data.append(sigma_col_avg)
    ppm_data.append(ppm_col_avg)
    uS_data.append(uS_col_avg)
    additional_data.append(additional_col_avg)

    # Initialize variables for storing encoded images
    encoded_sigma_image, encoded_ppm_image, encoded_uS_image, encoded_additional_image = None, None, None, None

    # Plotting tables with improved adjustments
    for data, title in [(sigma_data, "y Values at intersection"), (ppm_data, "BER PPM"), (uS_data, "BER PPM Values at windows = 2"), (additional_data, "y Values at windows = 2")]:
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.axis('off')
        #plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        column_widths = get_column_widths(data)
        table = ax.table(cellText=data, loc='center', colWidths=column_widths, cellLoc='center')
        table.set_fontsize(12)
        table.scale(1, 2)  # Adjust scale to enhance clarity
        plt.title(title, fontsize=12)
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        encoded_image = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
        
        if title == "y Values at intersection":
            encoded_sigma_image = encoded_image
        elif title == "BER PPM":
            encoded_ppm_image = encoded_image
        elif title == "BER PPM Values at windows = 2":
            encoded_uS_image = encoded_image
        else:
            encoded_additional_image = encoded_image

    return encoded_sigma_image, encoded_ppm_image, encoded_uS_image, encoded_additional_image

'''import csv
import io
def plot_ber_tables(ber_results, table_names):
    sigma_headers = ["State/Transition"] + [name for name in table_names] + ["Row Avg"]
    ppm_headers = ["State/Transition"] + [name for name in table_names] + ["Row Avg"]
    
    sigma_data = [sigma_headers]
    ppm_data = [ppm_headers]

    grouped_data = {}
    for entry in ber_results:
        key = entry[1]
        if key not in grouped_data:
            grouped_data[key] = []
        grouped_data[key].append((entry[2], entry[3]))

    for key, values in grouped_data.items():
        sigma_row = [key]
        ppm_row = [key]
        for sigma, ppm in values:
            sigma_row.append(f"{sigma:.4f}")
            ppm_row.append(f"{ppm:.0f}")
        sigma_row.append(f"{np.mean([float(val) for val in sigma_row[1:]]):.4f}")
        ppm_row.append(f"{np.mean([float(val) for val in ppm_row[1:]]):.0f}")
        sigma_data.append(sigma_row)
        ppm_data.append(ppm_row)

    sigma_col_avg = ["Col Avg"]
    ppm_col_avg = ["Col Avg"]
    for col in range(1, len(sigma_data[0])):
        sigma_col = [float(row[col]) for row in sigma_data[1:] if row[col] != "Col Avg"]
        ppm_col = [float(row[col]) for row in ppm_data[1:] if row[col] != "Col Avg"]
        sigma_col_avg.append(f"{np.mean(sigma_col):.4f}")
        ppm_col_avg.append(f"{np.mean(ppm_col):.0f}")
    sigma_data.append(sigma_col_avg)
    ppm_data.append(ppm_col_avg)

    # Generate CSV formatted string for sigma and ppm
    sigma_csv = io.StringIO()
    ppm_csv = io.StringIO()
    sigma_writer = csv.writer(sigma_csv)
    ppm_writer = csv.writer(ppm_csv)
    
    sigma_writer.writerows(sigma_data)
    ppm_writer.writerows(ppm_data)
    
    return sigma_csv.getvalue(), ppm_csv.getvalue()'''

def plot_combined_window_analysis_table(aggregated_window_values, figsize=(15, 10)):
    tables_order = sorted(set(table_name for (table_name, _), _ in aggregated_window_values.items()))

    # Initialize an empty list to maintain the order of state pairs based on their appearance
    state_pairs_order = []
    for _, pair in aggregated_window_values.keys():
        state_pair = f"State {pair[0]} & State {pair[1]}"
        if state_pair not in state_pairs_order:
            state_pairs_order.append(state_pair)

    # Initialize combined data with zeros for all state pairs across all tables
    combined_table_data = {state_pair: [0] * len(tables_order) for state_pair in state_pairs_order}

    # Populate the combined table data with actual values
    for (table_name, pair), value in aggregated_window_values.items():
        state_pair = f"State {pair[0]} & State {pair[1]}"
        table_index = tables_order.index(table_name)
        combined_table_data[state_pair][table_index] = value

    # Adding averages per state pair and per table
    for state_pair in combined_table_data:
        combined_table_data[state_pair].append(np.mean([float(v) for v in combined_table_data[state_pair]]))

    # Adding averages row at the end
    average_row = ['Average']
    for col in range(len(tables_order)):
        col_values = [float(combined_table_data[state_pair][col]) for state_pair in state_pairs_order]
        average_row.append(np.mean(col_values))
    average_row.append(np.mean(average_row[1:-1]))  # Calculate the overall average excluding the label

    combined_table_data['Average'] = average_row

    # Prepare table data with headers and rows
    header = ["State Pair"] + tables_order + ["Average"]
    table_data = [header]
    table_data.extend([[state_pair] + [f"{float(val):.2f}" for val in values] for state_pair, values in combined_table_data.items() if state_pair != 'Average'])
    # Ensure averages are also formatted to two decimal places
    table_data.append([average_row[0]] + [f"{val:.2f}" for val in average_row[1:]])

    # Plot combined table
    fig, ax = plt.subplots(figsize=figsize)
    #ax.axis('tight')
    ax.axis('off')
    column_widths = get_column_widths(table_data)
    table = ax.table(cellText=table_data, loc='center', colWidths=column_widths, cellLoc='center')
    #table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    ax.set_title('Window table 99% to 1%')
    #plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    encoded_image = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    return encoded_image

def calculate_window_values(groups, selected_groups):
    window_values_99_1 = {}
    window_values_999_01 = {}

    for i in range(len(selected_groups) - 1):
        group_id_a = selected_groups[i]
        group_id_b = selected_groups[i + 1]

        data_a = groups[group_id_a]
        data_b = groups[group_id_b]

        # Calculate percentiles for both scenarios
        percentile_99_a = np.percentile(data_a, 99)
        percentile_1_b = np.percentile(data_b, 1)
        percentile_999_a = np.percentile(data_a, 99.9)
        percentile_01_b = np.percentile(data_b, 0.1)

        # Calculate differences
        window_value_difference_99_1 = percentile_1_b - percentile_99_a
        window_value_difference_999_01 = percentile_01_b - percentile_999_a

        # Store results
        window_values_99_1[(group_id_a, group_id_b)] = window_value_difference_99_1
        window_values_999_01[(group_id_a, group_id_b)] = window_value_difference_999_01

    return window_values_99_1, window_values_999_01

def normalize_selected_groups(selected_groups):
    unique_sorted_elements = sorted(set(selected_groups))
    element_to_index = {element: index for index, element in enumerate(unique_sorted_elements)}

    # Also create a reverse mapping from normalized index to original group ID
    index_to_element = {index: element for index, element in enumerate(unique_sorted_elements)}

    normalized_groups = [element_to_index[element] for element in selected_groups]
    
    return normalized_groups, index_to_element

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
def get_colors(num_colors):
    """Generate a colormap and return the colors for the specified number of items."""
    cmap = plt.get_cmap('viridis', num_colors)
    norm = mcolors.Normalize(vmin=0, vmax=num_colors - 1)
    return [cmap(norm(i)) for i in range(num_colors)]

def plot_combined_window_analysis_table_2(aggregated_window_values, figsize=(12, 8)):
    # Initialize an empty list to maintain the order of state pairs based on their appearance
    state_pairs_order = []
    for pair in aggregated_window_values.keys():
        state_pair = f"State {pair[0]} & State {pair[1]}"
        if state_pair not in state_pairs_order:
            state_pairs_order.append(state_pair)

    # Initialize combined data with zeros for each state pair
    combined_table_data = {state_pair: [aggregated_window_values[pair]] for state_pair, pair in zip(state_pairs_order, aggregated_window_values.keys())}

    # Adding averages per state pair
    for state_pair, pair in zip(state_pairs_order, aggregated_window_values.keys()):
        combined_table_data[state_pair].append(np.mean([float(combined_table_data[state_pair][0])]))

    # Adding averages row at the end
    average_row = ['Average']
    col_values = [float(combined_table_data[state_pair][0]) for state_pair in state_pairs_order]
    average_row.append(np.mean(col_values))
    average_row.append(average_row[1])  # The overall average is the same since there's only one column

    combined_table_data['Average'] = average_row

    # Prepare table data with headers and rows
    header = ["State Pair", "Value", "Average"]
    table_data = [header]
    table_data.extend([[state_pair] + [f"{float(val):.2f}" for val in values] for state_pair, values in combined_table_data.items() if state_pair != 'Average'])
    table_data.append([average_row[0]] + [f"{val:.2f}" for val in average_row[1:]])

    # Plot combined table
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    #ax.set_title('Window table 99% to 1%')
    #plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    encoded_image = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    return encoded_image
