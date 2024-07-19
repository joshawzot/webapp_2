from tools_for_plots import *
import io
import base64
import pandas as pd

def reorder_tables(table_names):
    # Define the desired order
    desired_order = [
        'conductance_25c_csv',
        'conductance_55c_csv',
        'conductance_85c_csv',
        'conductance_125c_csv',
        'conductance_85c_2_csv',
        'conductance_55c_2_csv',
        'conductance_25c_2_csv'
    ]
    
    # Create a dictionary to map table names to their desired positions
    order_map = {name: i for i, name in enumerate(desired_order)}
    
    # Reorder according to the desired sequence
    reordered_tables = sorted(table_names, key=lambda name: order_map.get(name, float('inf')))
    
    return reordered_tables

def generate_plot(table_names, database_name, form_data):
    print("table_names:", table_names)
    #table_names = reorder_tables(table_names)
    #print("reordered_table_names:", table_names)
    selected_groups = form_data.get('selected_groups', [])
    print("selected_groups:", selected_groups)
    #print("len(selected_groups):", len(selected_groups))
    #selected_groups = list(range(4096))
    #selected_groups = list(range(82944))
    #return 0
    
    # Fetch sub_array_size from form_data, which could be either a string or a list
    sub_array_size_raw = form_data.get('sub_array_size', '324,64')  # Default to '324,64' if not present
    print("sub_array_size_raw:", sub_array_size_raw)
    sub_array_size = tuple(sub_array_size_raw)
    print("sub_array_size:", sub_array_size)

    # Initialize an empty list to hold the encoded plots
    encoded_plots = []
    group_data = []
    colors = get_colors(len(table_names))
    avg_values = []
    std_values = []
    
    for table_name in table_names:
        groups, stats, selected_groups = get_group_data_new(table_name, selected_groups, database_name, sub_array_size)

        # Extract average and standard deviation values for each selected group
        table_avg_values = [stat[2] for stat in stats]  # Index 2 is average
        table_std_values = [stat[3] for stat in stats]  # Index 3 is standard deviation

        group_data.append(groups)
        avg_values.append(table_avg_values)
        std_values.append(table_std_values)

        #window_values_99, window_values_999 = calculate_window_values(groups, selected_groups)
        #print("window_values_99:", window_values_99)

    #color map
    for table_name in table_names:
        data_matrix, data_matrix_size = get_full_table_data(table_name, database_name)
        encoded_plots.append(plot_colormap(data_matrix, title=f"Colormap for {table_name}"))

    # Generate plots for individual tables
    encoded_plots.append(plot_boxplot(group_data, table_names))
    #encoded_plots.append(plot_histogram(group_data, table_names, colors))

    encoded_plots.append(plot_average_values_table(avg_values, table_names, selected_groups))
    encoded_plots.append(plot_std_values_table(std_values, table_names, selected_groups))

    if len(selected_groups) != 1:
        plot_data_sigma, plot_data_cdf, plot_data_interpo, ber_results = plot_transformed_cdf_2(group_data, table_names, selected_groups, colors)
        encoded_plots.append(plot_data_sigma)
        encoded_plots.append(plot_data_cdf)
        #encoded_plots.append(plot_data_interpo)

        # Generate tables for visualizing BER results
        encoded_sigma_image, encoded_ppm_image, encoded_2uS_image, extra_image= plot_ber_tables(ber_results, table_names)
        #encoded_plots.append(encoded_sigma_image)
        encoded_plots.append(encoded_ppm_image)
        #encoded_plots.append(extra_image)
        #encoded_plots.append(encoded_2uS_image)

    return encoded_plots
