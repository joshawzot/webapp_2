import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from datetime import datetime
from io import BytesIO
import base64
import re
from tools_for_plots import (
    calculate_statistics,
    plot_boxplot,
    get_group_data_2,
    extract_date_time,
    calculate_windows_between_clusters,
    sort_table_names,
    plot_statistics_table,
    plot_transformed_cdf_figure,
    plot_cdf_figure,
    plot_histogram_density,
    plot_histogram_figure,
    plot_boxplot_figure
)

from datetime import datetime

def extract_datetime_from_table_name(table_name):
    # Assuming table_name format "YYYY_MM_DD_HHMMSS_X", where X can be any string
    datetime_part = table_name.split('_')[:4]  # Extracts the datetime components
    datetime_str = '_'.join(datetime_part)  # Rejoins the extracted parts into a string
    return datetime.strptime(datetime_str, "%Y_%m_%d_%H%M%S")

def sort_table_names(table_names):
    # Sort table names based on the datetime extracted from each name
    return sorted(table_names, key=extract_datetime_from_table_name)

def generate_plot_64x64(table_names, database_name, form_data):
    print('Processing Tables:')
    selected_groups = list(range(16))  # Selecting all groups from 0 to 15

    all_data = []
    all_stats = []
    tables_by_date = {}

    # Adjust the grouping of table names based on form_data
    date_option = form_data['date_option']

    if date_option == "all_date_together":
        # Treat all tables as belonging to a single date
        tables_by_date = {"all": table_names}
    elif date_option == "separate_every_file":
        # Treat each table as a separate date/group
        tables_by_date = {name: [name] for name in table_names}
    else:  # By date or any other case
        # Group tables by their actual dates
        table_names.sort(key=extract_date_time)
        tables_by_date = {}
        for table_name in table_names:
            date = extract_date_time(table_name).date()
            if date not in tables_by_date:
                tables_by_date[date] = []
            tables_by_date[date].append(table_name)

    if date_option == "separate_every_file":
        print("table_names before sorting:", table_names)
        table_names = sort_table_names(table_names)  # Make sure you've defined this function as shown earlier
        print("table_names after sorting:", table_names)

    new_combined_data = []
    new_combined_stats = []
    data_origin = []
    cluster_end_indices = []
    cumulative_cluster_count = 0

    # Initialize a counter for cluster numbering
    cluster_number = 1

    '''if date_option == "separate_every_file":
        print("table_names:", table_names)
        table_names = sort_table_names(table_names)
        print("table_names:", table_names)
        for table_name in table_names:
            groups, stats, origins = get_group_data_2(table_name, selected_groups, database_name)
            # Combine all group data into a single cluster representation
            cluster_data = np.concatenate(groups)
            new_combined_data.append(cluster_data)
            # Increment cluster_number once per table, after all its groups have been processed
            avg = round(np.mean(np.concatenate(groups)), 2)  # Example calculation for the whole table
            std = round(np.std(np.concatenate(groups)), 2)  # Example calculation for the whole table
            outlier_count = np.sum(np.abs(np.concatenate(groups) - avg) > round(2.698 * std, 2))
            outlier_percentage = round(outlier_count / len(np.concatenate(groups)) * 100, 2)
            new_combined_stats.append((f"Cluster {cluster_number}", avg, std, outlier_percentage))
            cluster_number += 1'''

    if date_option == "separate_every_file":
        print("table_names:", table_names)
        for table_name in table_names:
            groups, stats, origins = get_group_data_2(table_name, selected_groups, database_name)
            # Combine all group data into a single cluster representation
            cluster_data = np.concatenate(groups)
            new_combined_data.append(cluster_data)
            # Calculate statistics for the whole table
            avg = round(np.mean(np.concatenate(groups)), 2)  # Example calculation for the whole table
            std = round(np.std(np.concatenate(groups)), 2)  # Example calculation for the whole table
            outlier_count = np.sum(np.abs(np.concatenate(groups) - avg) > round(2.698 * std, 2))
            outlier_percentage = round(outlier_count / len(np.concatenate(groups)) * 100, 2)
            # Use table_name as the identifier instead of "Cluster X"
            new_combined_stats.append((f"{table_name}", avg, std, outlier_percentage))

    else:
        # Process each group of tables separated by dates
        for date, tables in tables_by_date.items():
            date_specific_data = []
            date_specific_stats = []

            for table_name in tables:
                groups, stats, origins = get_group_data_2(table_name, selected_groups, database_name)
                date_specific_data.extend(groups)
                date_specific_stats.extend(stats)
                data_origin.extend(origins)

            # Apply the clustering logic within each date group
            threshold_percentage = 10
            average_values = [stat[2] for stat in date_specific_stats]
            threshold = (max(average_values) - min(average_values)) * threshold_percentage / 100

            clusters = {}
            cluster_indices = []

            for i, avg1 in enumerate(average_values):
                if i not in cluster_indices:
                    cluster_indices.append(i)
                    current_cluster = [i]
                    for j, avg2 in enumerate(average_values):
                        if i != j and j not in cluster_indices and abs(avg1 - avg2) <= threshold:
                            current_cluster.append(j)
                            cluster_indices.append(j)
                    clusters[len(clusters) + 1] = current_cluster

            for cluster_index in sorted(clusters.keys(), key=lambda x: np.mean([average_values[i] for i in clusters[x]])):
                cluster_indices = clusters[cluster_index]
                cluster_data = [date_specific_data[i] for i in cluster_indices]
                if cluster_data:
                    combined_cluster_data = np.concatenate(cluster_data)
                    new_combined_data.append(combined_cluster_data)
                    avg = round(np.mean(combined_cluster_data), 2)
                    std = round(np.std(combined_cluster_data), 2)
                    outlier_count = np.sum(np.abs(combined_cluster_data - avg) > round(2.698 * std, 2))
                    outlier_percentage = round(outlier_count / len(combined_cluster_data) * 100, 2)

                    # Use the continuously incrementing cluster_number for labeling
                    new_combined_stats.append((f"Cluster {cluster_number}", avg, std, outlier_percentage))
                    cluster_number += 1

            # After processing each date group, update the cluster_end_indices
            cumulative_cluster_count += len(clusters)  # Increment by the number of clusters in this date group
            cluster_end_indices.append(cumulative_cluster_count - 1)  # Index of the last cluster for this date group

    print("cluster_number", cluster_number)
    #import sys
    #sys.exit()

    # Define a colormap
    cmap = cm.get_cmap('viridis', len(new_combined_data))
    norm = mcolors.Normalize(vmin=0, vmax=len(new_combined_data) - 1)
    colors = [cmap(norm(i)) for i in range(len(new_combined_data))]

    #print("Size of new_combined_data:", len(new_combined_data))

    # Before plotting, print the number of data points for each boxplot
    for i, cluster_data in enumerate(new_combined_data, start=1):
        print(f"Number of data points in Cluster {i}: {len(cluster_data)}")

    plot_operations = [
        lambda: plot_boxplot_figure(new_combined_data, new_combined_stats),
        lambda: calculate_windows_between_clusters(new_combined_data, data_origin, new_combined_stats, cluster_end_indices),
        #lambda: plot_histogram_figure(new_combined_data, colors),
        lambda: plot_histogram_density(new_combined_data, colors),
        #lambda: plot_cdf_figure(new_combined_data, colors),
        lambda: plot_transformed_cdf_figure(new_combined_data, colors),
        lambda: plot_statistics_table(new_combined_stats),
        # Add or comment out other plots or calculations as needed
    ]

    # Execute the plotting and calculation functions, collecting results
    plot_data = [operation() for operation in plot_operations if operation is not None]
    
    return plot_data

'''boxplot_data = plot_boxplot_figure(new_combined_data, new_combined_stats)
    histogram_data_uniform = plot_histogram_figure(new_combined_data, colors)
    histogram_data = plot_histogram_density(new_combined_data, colors)
    cdf_data = plot_cdf_figure(new_combined_data, colors)
    transformed_cdf_data = plot_transformed_cdf_figure(new_combined_data, colors)
    table_data = plot_statistics_table(new_combined_stats)
    window_table_data = calculate_windows_between_clusters(new_combined_data, data_origin, new_combined_stats, cluster_end_indices)
    
    # Close the connection
    close_connection()

    plot_data = [boxplot_data, window_table_data,
                histogram_data_uniform, histogram_data, cdf_data,
                transformed_cdf_data, table_data]

    return plot_data'''