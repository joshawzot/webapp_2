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
    
def sort_table_names(table_names):
    # Define a helper function to extract the numerical part from the table name
    def extract_number(name):
        # Find all occurrences of numbers in the string and return the first occurrence
        numbers = re.findall(r'\d+', name)
        return int(numbers[0]) if numbers else -1

    # Sort the list of table names using the extracted number as the key
    sorted_names = sorted(table_names, key=extract_number)
    return sorted_names

def calculate_windows_between_clusters(data_clusters, data_origin, stats, cluster_end_indices):
    windows_details = []
    start_idx = 0
    overlapping_details = []  # List to store overlapping details
    window_details = []  # List to store window details between clusters

    # Convert cluster_end_indices to a set for efficient lookup
    cluster_end_indices_set = set(cluster_end_indices)

    for i in range(len(data_clusters) - 1):
        # Skip overlap check for the last cluster of each date
        if i in cluster_end_indices_set:
            continue

        current_max = round(np.max(data_clusters[i]), 2)
        next_min = round(np.min(data_clusters[i + 1]), 2)
        window = round(next_min - current_max, 2)

        # Append window details and add 1 to the first and second column values
        window_details.append((i + 1, i + 2, current_max, next_min, window))

        overlapping_data = []

        if window < 0:
            # Overlapping clusters detected
            for data_point, origin in zip(data_clusters[i], data_origin[start_idx:start_idx+len(data_clusters[i])]):
                if data_point >= next_min:
                    overlapping_data.append((*origin, data_point))
                    overlapping_details.append(("Cluster " + str(i + 1), *origin, round(data_point, 2)))

            for data_point, origin in zip(data_clusters[i + 1], data_origin[start_idx+len(data_clusters[i]):start_idx+len(data_clusters[i])+len(data_clusters[i + 1])]):
                if data_point <= current_max:
                    overlapping_data.append((*origin, data_point))
                    overlapping_details.append(("Cluster " + str(i + 2), *origin, round(data_point, 2)))

        windows_details.append((window, current_max, next_min, overlapping_data))
        start_idx += len(data_clusters[i])

    # Create a table image for window details
    plt.figure(figsize=(12, 5))
    ax_table_window = plt.gca()
    ax_table_window.axis('on')
    col_labels_window = ["Cluster", "Next Cluster", "Current Max", "Next Min", "Window Value"]
    # Debug: Print the window_details to check their structure
    print(window_details)
    table_window = ax_table_window.table(cellText=window_details, colLabels=col_labels_window, loc='center')
    table_window.auto_set_font_size(False)
    table_window.set_fontsize(10)
    table_window.scale(1, 1.5)
    ax_table_window.axis('off')
    plt.tight_layout()

    # Save the Window Details Table figure
    buf_table_window = BytesIO()
    plt.savefig(buf_table_window, format='png', bbox_inches='tight')
    buf_table_window.seek(0)
    window_table_data = base64.b64encode(buf_table_window.getvalue()).decode('utf-8')
    buf_table_window.close()

    #return windows_details, overlapping_table_data, window_table_data
    #return windows_details, window_table_data
    return window_table_data

def extract_date_time(table_name):
    # Updated regex to match dates with hyphens or underscores
    datetime_match = re.search(r'^\d{4}[-_]\d{2}[-_]\d{2}', table_name)
    if datetime_match:
        datetime_part = datetime_match.group()
        # Replace underscore with hyphen if present
        datetime_part = datetime_part.replace('_', '-')
        return datetime.strptime(datetime_part, '%Y-%m-%d')
    else:
        raise ValueError("Invalid table name format")

def plot_boxplot(data, table_names, figsize=(15, 10)):
    plt.figure(figsize=figsize) 
    # Boxplot setup

    xticks = []
    xticklabels = table_names  # Adjusted to your groups' names
    
    print("data:", data)
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
    plt.ylabel('Conductance (u)', fontsize=12)
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
    
    plt.xlabel('Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    #plt.legend()
    plt.legend(loc='upper right')

    # Save the histogram figure to a buffer
    buf_histogram = BytesIO()
    plt.savefig(buf_histogram, format='png', bbox_inches='tight')
    buf_histogram.seek(0)
    plot_data_histogram = base64.b64encode(buf_histogram.getvalue()).decode('utf-8')
    buf_histogram.close()

    # Clear the figure to start fresh for another plot
    plt.clf()

    return plot_data_histogram

def plot_transformed_cdf(data, table_names, colors, figsize=(15, 10)):
    plt.figure(figsize=figsize)
    
    # Track the filenames that have been added to the legend
    added_to_legend = set()

    for i, group in enumerate(data):
        for j, subgroup in enumerate(group):
            # Only add the label the first time a filename is encountered
            label = f'{table_names[i]}' if table_names[i] not in added_to_legend else None
            if label:
                added_to_legend.add(table_names[i])
            
            # Sort the subgroup data
            sorted_data = np.sort(subgroup)
            # Calculate the CDF values and transform to sigma values
            #cdf_values = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            cdf_values = (np.arange(1, len(sorted_data) + 1) - 0.5) / len(sorted_data)
            sigma_values = sp_stats.norm.ppf(cdf_values)

            # Plot with the designated color and label (if applicable)
            plt.plot(sorted_data, sigma_values, linestyle='-', linewidth=1, color=colors[i], label=label)  # Thin lines
            plt.scatter(sorted_data, sigma_values, s=10, color=colors[i])  # Scatter dots

            # Plot with the designated color and label (if applicable)
            #plt.scatter(sorted_data, sigma_values, s=10, color=colors[i], label=label)

            # Plot with the designated color and label (if applicable)
            #plt.plot(sorted_data, sigma_values, linestyle='-', linewidth=1, color=colors[i], label=label)

    plt.xlabel('Transformed Data Value', fontsize=12)
    plt.ylabel('Sigma (Standard deviations)', fontsize=12)
    plt.title('Transformed CDF of Data by Groups')
    plt.legend()
    plt.grid(True)

    # Save and encode the figure
    buf_transformed_cdf = BytesIO()
    plt.savefig(buf_transformed_cdf, format='png', bbox_inches='tight')
    buf_transformed_cdf.seek(0)
    plot_data_transformed_cdf = base64.b64encode(buf_transformed_cdf.getvalue()).decode('utf-8')
    buf_transformed_cdf.close()
    plt.clf()  # Clear the figure for any future plotting

    return plot_data_transformed_cdf

def get_group_data_new(table_name, selected_groups, database_name, sub_array_size):
    connection = create_connection(database_name)
    query = f"SELECT * FROM `{table_name}`"
    cursor = connection.cursor()
    cursor.execute(query)
    data = cursor.fetchall()

    # Convert fetched data to a NumPy array for easier manipulation
    data_np = np.array(data)  # Assume the scaling factor was removed for clarity

    # Check if the average of data_np is more than 1, then scale data_np by multiplying it with 1e-6
    if np.mean(data_np) > 1:
        data_np = data_np * 1e-6

    groups = []
    groups_stats = []  # List to store statistics for each group

    rows_per_group, cols_per_group = sub_array_size
    total_rows, total_cols = data_np.shape
    print("total rows:", total_rows)
    print("total cols:", total_cols)

    num_row_groups = total_rows // rows_per_group
    num_col_groups = total_cols // cols_per_group
    num_of_groups = num_col_groups * num_row_groups
    partial_rows = total_rows % rows_per_group  # Check if there's a partial row group
    partial_cols = total_cols % cols_per_group  # Check if there's a partial column group

    group_idx = 0  # Initialize group index

    for i in range(num_row_groups + (1 if partial_rows > 0 else 0)):
        for j in range(num_col_groups + (1 if partial_cols > 0 else 0)):
            start_row = i * rows_per_group
            end_row = (i + 1) * rows_per_group if i < num_row_groups else total_rows

            start_col = j * cols_per_group
            end_col = (j + 1) * cols_per_group if j < num_col_groups else total_cols

            # Check if this group is selected
            if group_idx in selected_groups:
                print(f"Group {group_idx}: Start Row: {start_row}, End Row: {end_row}, Start Col: {start_col}, End Col: {end_col}")

                try:
                    group = data_np[start_row:end_row, start_col:end_col]
                    flattened_group = group.flatten()

                    # Filter out negative values
                    positive_flattened_group = flattened_group[flattened_group >= 0] * 1e6

                    groups.append(positive_flattened_group)

                    # Calculate statistics for the positive values
                    if len(positive_flattened_group) > 0:  # Ensure there are positive values to analyze
                        average = round(np.mean(positive_flattened_group), 2)
                        std_dev = round(np.std(positive_flattened_group), 2)
                        outlier_percentage = round(np.sum(np.abs(positive_flattened_group - average) > 2.698 * std_dev) / len(positive_flattened_group) * 100, 2)
                        groups_stats.append((table_name, group_idx, average, std_dev, outlier_percentage))
                    else:
                        print(f"Group {group_idx} has no positive values for analysis.")
                except IndexError as e:
                    print(f"Error accessing data slice: {e}")

            group_idx += 1  # Increment group index after each inner loop

    close_connection()
    
    #data_np.shape[1] means columns   #data_np.shape[0] means rows
    return groups, groups_stats, data_np.shape[1], num_of_groups

import itertools

''' Minimum Average Overlap: The primary criterion is to find the combination of groups that has the lowest average overlap among the sequential pairs within that combination. This determines which groups are considered best in terms of minimizing the intersection of data points.

    Maximum Average Gap: If there are combinations with the same average overlap, the next criterion is to select the combination that has the highest average gap. This seeks to maximize the average distance between the groups, aiming for a more evenly spaced distribution.

    Smallest Maximum-Minimum Gap Difference: If combinations still tie based on the average overlap and maximum average gap, the next criterion looks at the difference between the maximum gap and minimum gap within each combination. The aim is to select the combination with the smallest difference, which suggests a more uniform distribution of gaps between the groups.

    Smallest Statistical Gap (New Criterion): Finally, if all previous metrics are identical, the smallest statistical gap, based on the difference in the maximum values of consecutive groups, is used as a tiebreaker. This criterion ensures that the selected combination has the smallest difference between the maximum values of consecutive groups, further refining the selection to ensure minimal variation between groups.'''
def find_min_average_overlap(overlaps, group_stats):
    min_avg_overlap = float('inf') #positive infinity
    best_groups = []
    max_avg_gap = 0  # Initialize the maximum average gap
    min_smallest_stat_gap = float('inf')  # Initialize the smallest stat gap
    min_max_gap_minus_min_gap = float('inf')  # Difference between max and min gap in best case

    # Get all unique group indices from the keys of the overlap dictionary
    group_indices = set([key[0] for key in overlaps.keys()] + [key[1] for key in overlaps.keys()])

    # Generate all combinations of four distinct groups
    group_combinations = itertools.combinations(sorted(group_indices), 4)

    # Evaluate each combination of four groups
    for combination in group_combinations:
        # Get all sequential pairs from the current combination of four groups
        pairs = [(combination[i], combination[i + 1]) for i in range(len(combination) - 1)]
        overlap_values = []
        stat_gaps = []
        gaps = []

        # Calculate average overlap and stat gaps for the sequential pairs within the combination
        for pair in pairs:
            if pair in overlaps:
                overlap_values.append(overlaps[pair])
            elif (pair[1], pair[0]) in overlaps:  # Check both directions
                overlap_values.append(overlaps[(pair[1], pair[0])])
            # Calculate gaps using group_stats
            if pair[0] in group_stats and pair[1] in group_stats:
                stat_gaps.append(group_stats[pair[1]][0] - group_stats[pair[0]][0])  # max of next - max of previous
                gaps.append(pair[1] - pair[0])

        # Only calculate averages if all pairs have values and stats
        if len(overlap_values) == len(pairs) and len(stat_gaps) == len(pairs):
            avg_overlap = sum(overlap_values) / len(overlap_values)
            avg_gap = sum(gaps) / len(gaps)
            min_gap = min(gaps)
            max_gap = max(gaps)
            smallest_stat_gap = min(stat_gaps)
            max_gap_minus_min_gap = max_gap - min_gap

            # Update best groups based on new selection criteria
            #print("min_avg_overlap:", min_avg_overlap)
            if (avg_overlap < min_avg_overlap or
                (avg_overlap == min_avg_overlap and (avg_gap > max_avg_gap or
                (avg_gap == max_avg_gap and (max_gap_minus_min_gap < min_max_gap_minus_min_gap or
                (max_gap_minus_min_gap == min_max_gap_minus_min_gap and smallest_stat_gap < min_smallest_stat_gap)))))):
                min_avg_overlap = avg_overlap
                max_avg_gap = avg_gap
                min_max_gap_minus_min_gap = max_gap_minus_min_gap
                min_smallest_stat_gap = smallest_stat_gap
                best_groups = combination

    return best_groups, min_avg_overlap

def get_group_data_2(table_name, selected_groups, database_name):
    # Establish a connection to the database
    connection = create_connection(database_name)
    query = f"SELECT * FROM `{table_name}`"
    # Execute the query and fetch data
    cursor = connection.cursor()
    cursor.execute(query)
    data = cursor.fetchall()

    # Convert data to a 2D NumPy array if it's not already
    if not isinstance(data, np.ndarray):
        try:
            data = np.array(data)
        except Exception as e:
            print(f"Error converting data to NumPy array: {e}")
            return [], [], [], {}

    # Check the shape of the data
    if data.ndim != 2:
        print("Data is not in the expected 2D format.")
        return [], [], [], {}

    groups = []
    groups_stats = []  # List to store statistics for each group
    data_origin = []   # List to track the origin of each data point

    # Process data for each selected group
    for i in range(4):
        for j in range(4):
            idx = i*4 + j
            if idx in selected_groups:
                try:
                    group = data[i*16:(i+1)*16, j*16:(j+1)*16]
                    flattened_group = group.flatten() * 1e6
                    groups.append(flattened_group)

                    # Calculate statistics for each group
                    average = round(np.mean(flattened_group), 2)
                    std_dev = round(np.std(flattened_group), 2)
                    outlier_percentage = round(np.sum(np.abs(flattened_group - average) > 2.698 * std_dev) / len(flattened_group) * 100, 2)
                    groups_stats.append((table_name, idx, average, std_dev, outlier_percentage))

                    # Track origin for each data point in the group
                    for row in range(i*16, (i+1)*16):
                        for col in range(j*16, (j+1)*16):
                            data_origin.append((table_name, (row, col)))
                except IndexError as e:
                    print(f"Error accessing data slice: {e}")

    close_connection()

    return groups, groups_stats, data_origin

def plot_statistics_table(stats, figsize=(15, 10)):

    plt.figure(figsize=figsize)
    ax_table = plt.gca()
    ax_table.axis('on')
    col_labels = ["Cluster", "Average", "Std Dev", "Outlier %"]
    table = ax_table.table(cellText=stats, colLabels=col_labels, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax_table.axis('off')
    plt.tight_layout()

    buf_table = BytesIO()
    plt.savefig(buf_table, format='png', bbox_inches='tight')
    buf_table.seek(0)
    table_data = base64.b64encode(buf_table.getvalue()).decode('utf-8')
    buf_table.close()

    return table_data

def plot_transformed_cdf_figure(data, colors, figsize=(15, 10)):

    plt.figure(figsize=figsize)
    for i, group_data in enumerate(data):
        if len(group_data) < 2:
            print(f"Skipping Cluster {i+1} due to insufficient data (size: {len(group_data)})")
            continue

        sorted_data = np.sort(group_data)
        cdf_values = (np.arange(1, len(sorted_data) + 1) - 0.5) / len(sorted_data)
        sigma_values = sp_stats.norm.ppf(cdf_values)
        plt.scatter(sorted_data, sigma_values, color=colors[i], s=5, label=f'Cluster {i+1}')

    plt.xlabel('Transformed Data Value', fontsize=12)
    plt.ylabel('Sigma (Standard deviations)', fontsize=12)
    plt.title('Transformed CDF of Combined Data by Clusters')
    plt.ylim(-5, 5)
    plt.legend(loc='upper left')
    plt.tight_layout()

    buf_transformed_cdf = BytesIO()
    plt.savefig(buf_transformed_cdf, format='png', bbox_inches='tight')
    buf_transformed_cdf.seek(0)
    transformed_cdf_data = base64.b64encode(buf_transformed_cdf.getvalue()).decode('utf-8')
    buf_transformed_cdf.close()

    return transformed_cdf_data

def plot_cdf_figure(data, colors, figsize=(15, 10)):

    plt.figure(figsize=figsize)
    for i, group_data in enumerate(data):
        sorted_data = np.sort(group_data)
        yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)
        plt.plot(sorted_data, yvals, color=colors[i], label=f'Cluster {i+1}')

    plt.xlabel('Combined Data Value')
    plt.ylabel('CDF')
    plt.title('CDF of Combined Data by Clusters')
    plt.legend()
    plt.tight_layout()

    buf_cdf = BytesIO()
    plt.savefig(buf_cdf, format='png', bbox_inches='tight')
    buf_cdf.seek(0)
    cdf_data = base64.b64encode(buf_cdf.getvalue()).decode('utf-8')
    buf_cdf.close()

    return cdf_data

def plot_histogram_density(data, colors, figsize=(15, 10)):
    plt.figure(figsize=figsize)

    for i, cluster in enumerate(data):
        # Determine the range for each dataset
        min_value = min(cluster)
        max_value = max(cluster)

        # Create a list of bin edges with an increment of 1
        bin_edges = np.arange(min_value, max_value + 2, 1)

        plt.hist(cluster, bins=bin_edges, alpha=0.5, color=colors[i], label=f'Cluster {i+1}', density=True)

    plt.xlabel('Combined Data Value')
    plt.ylabel('Probability')
    plt.yscale('log')
    plt.title('Histogram of Combined Data by Clusters (Probability Density)')
    plt.legend()
    plt.tight_layout()

    # Save the histogram figure
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()

    return plot_data

def plot_histogram_figure(data, colors, figsize=(15, 10)):
    plt.figure(figsize=figsize)

    # Calculate the global min and max across all clusters to define a common bin range
    global_min = min([min(cluster) for cluster in data])
    global_max = max([max(cluster) for cluster in data])

    # Create bins with a width of 1, starting from global_min to global_max
    bins = np.arange(global_min, global_max + 1, 1)  # '+1' to include the rightmost edge

    for i, cluster_data in enumerate(data):
        # Use the predefined bins for each histogram
        plt.hist(cluster_data, bins=bins, alpha=0.5, color=colors[i], label=f'Cluster {i+1}')

    plt.xlabel('Combined Data Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Combined Data by Clusters with Uniform Bin Width')
    plt.legend()
    plt.tight_layout()

    # Save the histogram figure with uniform bin width
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    histogram_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()

    return histogram_data

def plot_boxplot_figure(data, stats, figsize=(15, 10)):
    plt.figure(figsize=figsize)
    plt.boxplot(data)
    plt.xticks(range(1, len(stats) + 1), ['Cluster ' + str(i) for i in range(1, len(stats) + 1)])
    plt.ylabel('Combined Data Value')
    plt.title('Boxplot of Combined Data by Clusters')
    plt.grid(True)
    plt.tight_layout()

    # Save the boxplot figure
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()

    return plot_data

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

def plot_overlap_table(combined_overlaps, table_names, selected_groups, data_matrix_size, num_of_groups):
    # Unpack data_matrix_size into a and b
    a, b = data_matrix_size

    # Helper function to create a figure with table
    def create_table_figure(table_data, column_widths, title):
        fig, ax = plt.subplots(figsize=(12, 8))  # Adjust the size as needed
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=table_data, loc='center', colWidths=column_widths, cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.2)
        ax.set_title(title)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        encoded_image = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return encoded_image

    header = ["State"] + [f"{table_name}" for table_name in table_names] + ["Average"]
    table_data1, table_data2 = [header], [header]  # Initialize table data lists with headers

    # Calculate the data for the tables
    num_group_pairs = len(selected_groups) - 1
    for i in range(num_group_pairs):
        row1 = [f"State {selected_groups[i]} & State {selected_groups[i+1]}"]
        row2 = [f"State {selected_groups[i]} & State {selected_groups[i+1]}"]
        valid_counts1 = []
        valid_counts2 = []

        for j, overlaps in enumerate(combined_overlaps):
            if i < len(overlaps):
                overlap_count1 = overlaps[i][2]
                overlap_count2 = overlaps[i][3]  # Assuming this index exists and is valid
                row1.append(f"{overlap_count1}")
                row2.append(f"{overlap_count2:.4f}%")
                valid_counts1.append(overlap_count1)
                valid_counts2.append(overlap_count2)
            else:
                row1.append("-")
                row2.append("-")

        average_count1 = sum(valid_counts1) / len(valid_counts1) if valid_counts1 else 0
        average_count2 = sum(valid_counts2) / len(valid_counts2) if valid_counts2 else 0
        row1.append(f"{average_count1:.4f}")
        row2.append(f"{average_count2:.4f}%")
        table_data1.append(row1)
        table_data2.append(row2)

    # Calculate averages for each column and add a final "Average" row
    averages_row1 = ["Average"]
    averages_row2 = ["Average"]
    for col in range(1, len(header)):  # Skip the first column "State"
        column_values1 = [float(row[col].replace('%', '')) if '%' in row[col] else float(row[col]) for row in table_data1[1:] if row[col] != "-"]
        column_values2 = [float(row[col].replace('%', '')) if '%' in row[col] else float(row[col]) for row in table_data2[1:] if row[col] != "-"]
        averages_row1.append(f"{sum(column_values1) / len(column_values1) if column_values1 else 0:.4f}")
        averages_row2.append(f"{sum(column_values2) / len(column_values2) if column_values2 else 0:.4f}%")

    table_data1.append(averages_row1)
    table_data2.append(averages_row2)

    # Assuming get_column_widths is defined elsewhere and calculates column widths
    column_widths1 = get_column_widths(table_data1)
    column_widths2 = get_column_widths(table_data2)

    # Create tables and encode as images
    encoded_image1 = create_table_figure(table_data1, column_widths1, "Overlap count")
    encoded_image2 = create_table_figure(table_data2, column_widths2, "Overlap percentage")

    return encoded_image1, encoded_image2

def plot_average_values_table(avg_values, table_names, selected_groups, figsize=(12, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('tight')
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
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)

    plt.title('Average table')
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    encoded_image = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return encoded_image

def plot_std_values_table(std_values, table_names, selected_groups, figsize=(12, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('tight')
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
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)
    
    plt.title('Sigma table')
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    encoded_image = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return encoded_image

def plot_colormap(data, title, figsize=(10, 8)):
    g_range=[0, 200]
    fig, ax = plt.subplots(figsize=figsize)
    if g_range:
        cax = ax.imshow(data * 1e6, cmap=plt.cm.viridis, origin="lower", vmin=g_range[0], vmax=g_range[1])
    else:
        cax = ax.imshow(data * 1e6, cmap=plt.cm.viridis, origin="lower")

    fig.colorbar(cax)
    
    ax.set_title(title)
    
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
    
    # Check if the average of data_matrix is more than 1, then scale data_matrix by multiplying it with 1e-6
    if np.mean(data_matrix) > 1:
        data_matrix = data_matrix * 1e-6
    
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
    table.scale(1, 1.2)
    ax.set_title('Combined Overlap for All Tables')

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    encoded_image = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    print("Plot generated.")
    return encoded_image

def plot_distributions(fit_params):
    plt.figure(figsize=(12, 8))

    for idx, info in fit_params.items():
        dist_name = info['distribution']
        params = info['parameters']
        dist = getattr(sp_stats, dist_name)
        
        # Define the lower and upper bounds using the PPF to ensure tails are included
        # Lower bound set to the PPF at a low probability (0.001 or lower)
        # Upper bound set to the PPF at a high probability (0.999 or higher)
        low_prob = 0.0000001 if dist_name == 'expon' else 0.0000001
        high_prob = 0.999999 if dist_name == 'expon' else 0.9999999
        
        lower_bound = dist.ppf(low_prob, *params[:-2], loc=params[-2], scale=params[-1])
        upper_bound = dist.ppf(high_prob, *params[:-2], loc=params[-2], scale=params[-1])
        
        x = np.linspace(lower_bound, upper_bound, 1000)
        pdf = dist.pdf(x, *params)
        plt.plot(x, pdf, label=f'{dist_name.capitalize()} Dist {idx}')

    plt.title('PDFs of Various Distributions')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    
    # Save plot to a bytes buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Encode plot to base64 string and return
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close()  # Clear the current figure to free memory
    return encoded

'''calculate not only the number of elements in group (n+1) that are smaller than the maximum of group (n), but also the number of elements in group (n) that are larger than the minimum of group (n+1). Then, for each pair of groups, you will choose the smaller count from these two calculations to determine the overlap.
'''
def calculate_overlap(group_data, selected_groups, sub_array_size):
    c, d = sub_array_size
    overlaps = []
    group_stats = {}  # Dictionary to store max and min of each group

    # Create a mapping of selected group indices to their corresponding data in groups
    group_mapping = {idx: data for idx, data in zip(selected_groups, group_data)}

    for i in range(len(selected_groups) - 1):
        group1 = group_mapping[selected_groups[i]]
        group2 = group_mapping[selected_groups[i+1]]

        # Calculate max and min for each group and store it
        max_value_group1 = np.max(group1)
        min_value_group1 = np.min(group1)
        max_value_group2 = np.max(group2)
        min_value_group2 = np.min(group2)

        group_stats[selected_groups[i]] = (max_value_group1, min_value_group1)
        if i == len(selected_groups) - 2:  # Ensure the last group's stats are also added
            group_stats[selected_groups[i+1]] = (max_value_group2, min_value_group2)

        # Calculate overlaps according to new definitions
        overlapping_values1 = group2[group2 < max_value_group1]
        overlapping_values2 = group1[group1 > min_value_group2]

        overlap_count1 = len(overlapping_values1)
        overlap_count2 = len(overlapping_values2)
        chosen_overlap_count = min(overlap_count1, overlap_count2)
        chosen_overlap_percentage = chosen_overlap_count / (c * d) * 100

        overlaps.append((selected_groups[i], selected_groups[i+1], chosen_overlap_count, chosen_overlap_percentage))
        
        '''print(f"Max of group {selected_groups[i]}:", max_value_group1)  
        print(f"Min of group {selected_groups[i]}:", min_value_group1)
        print(f"Max of group {selected_groups[i+1]}:", max_value_group2)
        print(f"Min of group {selected_groups[i+1]}:", min_value_group2)
        print("group stats:", group_stats) #max and min of each group
        print(f"Overlap count (smaller values in (n+1)): {overlap_count1}")
        print(f"Overlap count (larger values in (n)): {overlap_count2}")
        print(f"Chosen overlap count: {chosen_overlap_count}")'''
        
    return overlaps, group_stats

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

def calculate_all_pairs_overlap(groups, sub_array_size):
    """Calculate overlaps for all pairs of groups with both conditions."""
    c, d = sub_array_size
    overlaps = {}
    group_stats = {}  # Dictionary to store max and min of each group
    group_indices = list(groups.keys())

    # Calculate and store max and min for each group
    for idx in group_indices:
        group = groups[idx]
        max_value = np.max(group)
        min_value = np.min(group)
        group_stats[idx] = (max_value, min_value)

    # Calculate overlaps for each pair of groups
    for i in range(len(group_indices)):
        for j in range(i + 1, len(group_indices)):
            group1_idx = group_indices[i]
            group2_idx = group_indices[j]
            group1 = groups[group1_idx]
            group2 = groups[group2_idx]

            max_value_group1 = group_stats[group1_idx][0]
            min_value_group2 = group_stats[group2_idx][1]

            # Find elements in group2 less than max of group1 and elements in group1 greater than min of group2
            overlapping_values1 = group2[group2 < max_value_group1]
            overlapping_values2 = group1[group1 > min_value_group2]

            overlap_count1 = len(overlapping_values1)
            overlap_count2 = len(overlapping_values2)

            # Choose the minimum overlap count to determine the overlap percentage
            chosen_overlap_count = min(overlap_count1, overlap_count2)
            chosen_overlap_percentage = chosen_overlap_count / (c * d) * 100

            # Store the chosen overlap percentage
            overlaps[(group1_idx, group2_idx)] = chosen_overlap_percentage

    #return overlaps, group_stats
    return overlaps

def generate_overlap_data_for_all_combinations(groups, selected_groups, sub_array_size):
    """Generate overlaps for all combinations of four selected groups."""
    group_data = {idx: groups[idx] for idx in selected_groups}
    all_pair_overlaps = calculate_all_pairs_overlap(group_data, sub_array_size)

    return all_pair_overlaps

def plot_min_4level_table(best_groups_append, min_overlap_append, table_names, figsize=(12, 8)):
    print("Generating best group overlap plot...")
    encoded_images = []

    # Debug: Check input data
    print("Best groups by table:", best_groups_append)
    print("Minimum overlaps by table:", min_overlap_append)

    # Prepare the data for the table
    header = ["Table Name", "Best States", "Average Overlap"]
    table_data = [header]

    # Data for individual tables
    for name, groups, overlap in zip(table_names, best_groups_append, min_overlap_append):
        table_data.append([name, ', '.join(map(str, groups)), f"{overlap:.4f}%"])

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('tight')
    ax.axis('off')
    
    # Create and adjust the table
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)  # Adjust table size

    ax.set_title('Best 4 States with Minimum Average Overlap')
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    # Save to buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    encoded_image = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    return encoded_image