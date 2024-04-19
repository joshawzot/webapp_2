from tools_for_plots import *
import io
import base64
import pandas as pd

def generate_plot(table_names, database_name, form_data):
    # Ensure this connects to your database
    connection = create_connection(database_name)

    selected_groups = form_data.get('selected_groups', [])

    # Fetch sub_array_size from form_data, which could be either a string or a list
    sub_array_size_raw = form_data.get('sub_array_size', '324,64')  # Default to '324,64' if not present
    print("sub_array_size_raw:", sub_array_size_raw)
    sub_array_size = tuple(sub_array_size_raw)

    # Initialize an empty list to hold the encoded plots
    encoded_plots = []
    group_data = []
    all_ppm_overlaps = []  # To collect all ppm overlaps

    # Define a colormap
    cmap = cm.get_cmap('viridis', len(table_names))
    norm = mcolors.Normalize(vmin=0, vmax=len(table_names) - 1)
    colors = [cmap(norm(i)) for i in range(len(table_names))]

    avg_values = []
    std_values = []
    all_groups = []
    best_groups_append = []
    min_overlap_append = []

    # Process each table and collect data and statistics (only once)
    for table_name in table_names:
        groups, stats, _, num_of_groups = get_group_data_new(table_name, selected_groups, database_name, sub_array_size)

        group_data.append(groups)
        all_groups.extend(groups)  # Extend the all_groups list with groups from each table

        # Calculate PPM overlaps for each individual table
        ppm_overlaps = calculate_overlap(groups, selected_groups, sub_array_size)
        all_ppm_overlaps.append(ppm_overlaps)  # Collecting ppm overlaps for all tables including individual ones

        # Extract average and standard deviation values for each selected group
        table_avg_values = [stat[2] for stat in stats]  # Index 2 is average
        table_std_values = [stat[3] for stat in stats]  # Index 3 is standard deviation

        avg_values.append(table_avg_values)
        std_values.append(table_std_values)

        #if selected_groups == [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]:
        if selected_groups == list(range(16)):
            #print(groups)
            complete_ppm_overlaps_individule = generate_overlap_data_for_all_combinations(groups, selected_groups, sub_array_size)
            print("complete_ppm_overlaps:", complete_ppm_overlaps_individule)
            best_groups, min_overlap = find_min_average_overlap(complete_ppm_overlaps_individule)
            print("Best 4 groups with minimum average overlap:", best_groups)
            print("Minimum average overlap:", min_overlap)
            
            best_groups_append.append(best_groups)
            min_overlap_append.append(min_overlap)

    # After all tables are processed, compute overlaps and find best groups based on all accumulated data
    #if selected_groups == [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]:
    if selected_groups == list(range(16)):
        complete_ppm_overlaps = generate_overlap_data_for_all_combinations(all_groups, selected_groups, sub_array_size)
        print("complete_ppm_overlaps:", complete_ppm_overlaps)
        best_groups_all, min_overlap_all = find_min_average_overlap(complete_ppm_overlaps)
        print("Best 4 groups with minimum average overlap:", best_groups_all)
        print("Minimum average overlap:", min_overlap_all)
    
        encoded_plots.append(plot_min_4level_table(best_groups_all, min_overlap_all, best_groups_append, min_overlap_append, table_names))

    # Within your main plotting function or loop
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

    encoded_plots.append(plot_transformed_cdf(group_data, table_names, colors))
    encoded_plots.append(plot_overlap_table(all_ppm_overlaps, table_names, selected_groups, data_matrix_size, num_of_groups))

    encoded_plot_for_avg = plot_average_values_table(avg_values, table_names, selected_groups)
    encoded_plots.append(encoded_plot_for_avg)

    encoded_plot_for_std = plot_std_values_table(std_values, table_names, selected_groups)
    encoded_plots.append(encoded_plot_for_std)

    return encoded_plots

