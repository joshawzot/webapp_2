from tools_for_plots import *
import io
import base64
import pandas as pd

#encoded_image1, encoded_image2 = plot_window_analysis_table(aggregated_groups_dict, selected_groups)
'''
def plot_window_analysis_table(group_data, selected_groups, figsize=(12, 8)):
    """
    group_data is a dictionary where keys are group IDs and values are lists or NumPy arrays of data points for each group.
    """
    window_analysis_data_99_1 = {}
    window_analysis_data_9999_001 = {}

    # Iterate over the selected groups to calculate window values
    for i in range(len(selected_groups) - 1):
        state_a = selected_groups[i]
        state_b = selected_groups[i + 1]
        state_pair = f"State {state_a} & State {state_b}"

        # Fetch data for each state and ignore all zeros
        data_a = np.array(group_data[state_a])[np.array(group_data[state_a]) != 0]
        data_b = np.array(group_data[state_b])[np.array(group_data[state_b]) != 0]

        # Calculate the 99th and 1st percentiles
        percentile_99_a = np.percentile(data_a, 99)
        percentile_1_b = np.percentile(data_b, 1)
        window_analysis_data_99_1[state_pair] = percentile_1_b - percentile_99_a

        # Calculate the 99.99th and 0.01st percentiles
        percentile_9999_a = np.percentile(data_a, 99.99)
        percentile_001_b = np.percentile(data_b, 0.01)
        window_analysis_data_9999_001[state_pair] = percentile_001_b - percentile_9999_a

    # Generate tables and encode images for both datasets
    encoded_images = []
    for data, title in [(window_analysis_data_99_1, "Window Analysis (99% - 1%)"), (window_analysis_data_9999_001, "Window Analysis (99.99% - 0.01%)")]:
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis('tight')
        ax.axis('off')
        header = ["State Pair", "Window Value"]
        table_data = [header] + [[pair, f"{vals:.2f}"] for pair, vals in data.items()]

        if len(table_data) > 1:
            table = ax.table(cellText=table_data, loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.2)
            ax.set_title(title)
        else:
            ax.text(0.5, 0.5, "No window analysis data available", va='center', ha='center')

        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        encoded_image = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        encoded_images.append(encoded_image)

    return encoded_images[0], encoded_images[1]
'''
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
    all_overlaps = []  # To collect all overlaps

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

        # Calculate overlaps for each individual table
        overlaps, group_stats = calculate_overlap(groups, selected_groups, sub_array_size)
        all_overlaps.append(overlaps)  # Collecting overlaps for all tables including individual ones

        # Extract average and standard deviation values for each selected group
        table_avg_values = [stat[2] for stat in stats]  # Index 2 is average
        table_std_values = [stat[3] for stat in stats]  # Index 3 is standard deviation

        avg_values.append(table_avg_values)
        std_values.append(table_std_values)

        #if selected_groups == [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]:
        if selected_groups == list(range(16)):
            #print(groups)
            complete_overlaps_individule = generate_overlap_data_for_all_combinations(groups, selected_groups, sub_array_size)
            #print("complete_overlaps:", complete_overlaps_individule)
            best_groups, min_overlap = find_min_average_overlap(complete_overlaps_individule, group_stats)
            #print("Best 4 groups with minimum average overlap:", best_groups)
            #print("Minimum average overlap:", min_overlap)
            
            best_groups_append.append(best_groups)
            min_overlap_append.append(min_overlap)

    # After all tables are processed, compute overlaps and find best groups based on all accumulated data
    #if selected_groups == [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]:
    if selected_groups == list(range(16)):
        complete_overlaps = generate_overlap_data_for_all_combinations(all_groups, selected_groups, sub_array_size)
        print("complete_overlaps:", complete_overlaps)
        best_groups_all, min_overlap_all = find_min_average_overlap(complete_overlaps, group_stats)
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

    encoded_image1, encoded_image2 = plot_overlap_table(all_overlaps, table_names, selected_groups, data_matrix_size, num_of_groups)
    encoded_plots.append(encoded_image1)
    encoded_plots.append(encoded_image2)

    encoded_plot_for_avg = plot_average_values_table(avg_values, table_names, selected_groups)
    encoded_plots.append(encoded_plot_for_avg)

    encoded_plot_for_std = plot_std_values_table(std_values, table_names, selected_groups)
    encoded_plots.append(encoded_plot_for_std)

    return encoded_plots

