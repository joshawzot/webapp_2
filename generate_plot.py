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
    aggregated_groups_by_selected_group = {group: [] for group in form_data.get('selected_groups', [])}
    print("aggregated_groups_by_selected_group:", aggregated_groups_by_selected_group)
    selected_groups = form_data.get('selected_groups', [])
    print("selected_groups:", selected_groups)
    print("len(selected_groups):", len(selected_groups))
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
    all_overlaps = []  # To collect all overlaps
    colors = get_colors(len(table_names))
    avg_values = []
    std_values = []
    all_groups = []
    best_groups_append = []
    min_overlap_append = []

    aggregated_window_values = {}
    
    # Process each table and collect data and statistics (only once)
    for table_name in table_names:
        groups, stats, _, num_of_groups, selected_groups = get_group_data_new(table_name, selected_groups, database_name, sub_array_size) #real selected_groups
        print("selected_groups_original:", selected_groups)
        
        # Extract average and standard deviation values for each selected group
        table_avg_values = [stat[2] for stat in stats]  # Index 2 is average
        table_std_values = [stat[3] for stat in stats]  # Index 3 is standard deviation

        # Sort groups based on average values in ascending order
        sorted_indices = sorted(range(len(table_avg_values)), key=lambda x: table_avg_values[x])
        groups = [groups[i] for i in sorted_indices]
        table_avg_values = [table_avg_values[i] for i in sorted_indices]
        table_std_values = [table_std_values[i] for i in sorted_indices]
        stats = [stats[i] for i in sorted_indices]
        #selected_groups = [selected_groups[i] for i in sorted_indices]
        #print("selected_groups_after_sorting:", selected_groups)

        # Normalize and process the selected groups after sorting
        normalized_groups, index_to_element = normalize_selected_groups(selected_groups)
        print("normalized_groups:", normalized_groups)
        print(f"Original: {selected_groups}, Normalized: {normalized_groups}")

        if len(selected_groups) != 1:
            window_values_99, window_values_999 = calculate_window_values(groups, normalized_groups)

            for pair, value in window_values_99.items():
                # Map normalized group pairs back to original group IDs
                original_pair = (index_to_element[pair[0]], index_to_element[pair[1]])
                aggregated_window_values[(table_name, original_pair)] = value
            
            # Calculate overlaps for each individual table
            overlaps, group_stats = calculate_overlap(groups, selected_groups, sub_array_size)
            all_overlaps.append(overlaps)  # Collecting overlaps for all tables including individual ones

        group_data.append(groups)
        all_groups.extend(groups)

        avg_values.append(table_avg_values)
        std_values.append(table_std_values)

        '''if selected_groups == list(range(16)): #[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]:
            #print(groups)
            complete_overlaps_individule = generate_overlap_data_for_all_combinations(groups, selected_groups, sub_array_size)
            print("complete_overlaps_individule:", complete_overlaps_individule)
            best_groups, min_overlap = find_min_average_overlap(complete_overlaps_individule, group_stats)
            print("Best 4 groups with minimum average overlap:", best_groups)
            print("Minimum average overlap:", min_overlap)
            
            best_groups_append.append(best_groups)
            min_overlap_append.append(min_overlap)'''

    #print_average_values_table(avg_values, table_names, selected_groups)
    #return avg_values, std_values, table_names, selected_groups
    #return 0
    #print("group_data:", group_data)
    #print("all_groups:", all_groups)

    #print("aggregated_window_values:", aggregated_window_values)

    #if selected_groups == list(range(16)):    
        #encoded_plots.append(plot_min_4level_table(best_groups_append, min_overlap_append, table_names))

    #color map
    for table_name in table_names:
        # Retrieve the full 2D data matrix for the current table_name
        data_matrix, data_matrix_size = get_full_table_data(table_name, database_name)
        
        # Plot the colormap for the retrieved data_matrix
        colormap_image = plot_colormap(data_matrix, title=f"Colormap for {table_name}")
        
        # Add the generated colormap image (encoded) to your list of encoded_plots
        encoded_plots.append(colormap_image)

    # Generate plots for individual tables
    encoded_plots.append(plot_boxplot(group_data, table_names))
    encoded_plots.append(plot_histogram(group_data, table_names, colors))
    #encoded_plots.append(plot_transformed_cdf(group_data, table_names, colors))
    #encoded_plots.append(plot_transformed_cdf_2(group_data, table_names, colors)[0])

    encoded_plot_for_avg = plot_average_values_table(avg_values, table_names, selected_groups)
    encoded_plots.append(encoded_plot_for_avg)

    encoded_plot_for_std = plot_std_values_table(std_values, table_names, selected_groups)
    encoded_plots.append(encoded_plot_for_std)

    if len(selected_groups) != 1:
        plot_data_sigma, plot_data_cdf, plot_data_interpo, ber_results = plot_transformed_cdf_2(group_data, table_names, selected_groups, colors)
        encoded_plots.append(plot_data_sigma)
        encoded_plots.append(plot_data_cdf)
        encoded_plots.append(plot_data_interpo)

        #table = plot_2uS_table(ppm, selected_groups)
        #encoded_plots.append(table)

        # Generate tables for visualizing BER results
        print("ber_results:", ber_results)
        encoded_sigma_image, encoded_ppm_image, encoded_2uS_image, extra_image= plot_ber_tables(ber_results, table_names)
        #return ber_results, table_names
        encoded_plots.append(encoded_sigma_image)
        encoded_plots.append(encoded_ppm_image)
        encoded_plots.append(extra_image)
        encoded_plots.append(encoded_2uS_image)

        #encoded_image1, encoded_image2 = plot_overlap_table(all_overlaps, table_names, selected_groups, data_matrix_size, num_of_groups)
        #encoded_plots.append(encoded_image1)
        #encoded_plots.append(encoded_image2)

        # Generate and append a single combined window analysis table plot
        encoded_window_analysis_plot = plot_combined_window_analysis_table(aggregated_window_values)
        #return aggregated_window_values
        encoded_plots.append(encoded_window_analysis_plot)

    return encoded_plots
