from tools_for_plots import *
import io
import base64
import pandas as pd

def plot_combined_window_analysis_table(aggregated_window_values, figsize=(12, 8)):
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
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)
    ax.set_title('Window table 99% to 1%')
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    encoded_image = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    return encoded_image

'''The function plot_window_analysis_table calculates the percentiles based on the raw data points from the provided group_data.'''
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

def calculate_window_values(groups, selected_groups):
    """
    Calculate the window value differences based on the selected groups for one table.
    Assumes `groups` is a dictionary where keys are group IDs and values are lists or NumPy arrays of data points for each group.
    """
    #print("groups:", groups)
    window_values = {}
    for i in range(len(selected_groups) - 1):
        group_id_a = selected_groups[i]
        group_id_b = selected_groups[i + 1]

        # Debug: Print the current groups being processed
        print(f"Processing groups: {group_id_a} and {group_id_b}")

        data_a = groups[group_id_a]
        data_b = groups[group_id_b]

        # Debug: Print sizes of the groups to ensure they're being accessed correctly
        print(f"Size of group {group_id_a}: {len(data_a)}, Size of group {group_id_b}: {len(data_b)}")

        percentile_99_a = np.percentile(data_a, 99)
        percentile_1_b = np.percentile(data_b, 1)

        # Debug: Print the calculated percentiles to verify correctness
        print(f"99th percentile of group {group_id_a}: {percentile_99_a}")
        print(f"1st percentile of group {group_id_b}: {percentile_1_b}")

        window_value_difference =  percentile_1_b - percentile_99_a

        # Debug: Print the window value difference
        print(f"Window value difference between group {group_id_a} and {group_id_b}: {window_value_difference}")

        window_values[(group_id_a, group_id_b)] = window_value_difference

    # Debug: Print final window values dictionary
    print("Final window values:", window_values)
    
    return window_values

def normalize_selected_groups(selected_groups):
    unique_sorted_elements = sorted(set(selected_groups))
    element_to_index = {element: index for index, element in enumerate(unique_sorted_elements)}

    # Also create a reverse mapping from normalized index to original group ID
    index_to_element = {index: element for index, element in enumerate(unique_sorted_elements)}

    normalized_groups = [element_to_index[element] for element in selected_groups]
    
    return normalized_groups, index_to_element

def get_colors(num_colors):
    """Generate a colormap and return the colors for the specified number of items."""
    cmap = cm.get_cmap('viridis', num_colors)
    norm = mcolors.Normalize(vmin=0, vmax=num_colors - 1)
    return [cmap(norm(i)) for i in range(num_colors)]

def generate_plot(table_names, form_data):
    print("table_names:", table_names)
    aggregated_groups_by_selected_group = {group: [] for group in form_data.get('selected_groups', [])}
    print("aggregated_groups_by_selected_group:", aggregated_groups_by_selected_group)
    selected_groups = form_data.get('selected_groups', [])
    print("selected_groups:", selected_groups)
    
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
        groups, stats, _, num_of_groups, selected_groups = get_group_data_new(table_name, selected_groups, sub_array_size) #real selected_groups
        print("selected_groups:", selected_groups)
        #print("groups:", groups)
        #print("stats:", stats)
        #print("num_of_groups:", num_of_groups)

        normalized_groups, index_to_element = normalize_selected_groups(selected_groups)
        print("normalized_groups:", normalized_groups)
        print(f"Original: {selected_groups}, Normalized: {normalized_groups}")
        window_values = calculate_window_values(groups, normalized_groups)

        for pair, value in window_values.items():
            # Map normalized group pairs back to original group IDs
            original_pair = (index_to_element[pair[0]], index_to_element[pair[1]])
            aggregated_window_values[(table_name, original_pair)] = value
            
        group_data.append(groups)
        all_groups.extend(groups)

        # Calculate overlaps for each individual table
        #overlaps, group_stats = calculate_overlap(groups, selected_groups, sub_array_size)
        #all_overlaps.append(overlaps)  # Collecting overlaps for all tables including individual ones

        # Extract average and standard deviation values for each selected group
        table_avg_values = [stat[2] for stat in stats]  # Index 2 is average
        table_std_values = [stat[3] for stat in stats]  # Index 3 is standard deviation

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

    print("group_data:", group_data)
    #print("all_groups:", all_groups)

    print("aggregated_window_values:", aggregated_window_values)

    '''if selected_groups == list(range(16)):    
        encoded_plots.append(plot_min_4level_table(best_groups_append, min_overlap_append, table_names))'''

    #color map
    #for table_name in table_names:
        # Retrieve the full 2D data matrix for the current table_name
        #data_matrix, data_matrix_size = get_full_table_data(table_name, database_name)
        
        # Plot the colormap for the retrieved data_matrix
        # colormap_image = plot_colormap(data_matrix, title=f"Colormap for {table_name}")
        
        # Add the generated colormap image (encoded) to your list of encoded_plots
        # encoded_plots.append(colormap_image)

    # Generate plots for individual tables
    encoded_plots.append(plot_boxplot(group_data, table_names))
    #encoded_plots.append(plot_histogram(group_data, table_names, colors))

    #encoded_plots.append(plot_transformed_cdf(group_data, table_names, colors))
    #encoded_plots.append(plot_transformed_cdf_2(group_data, table_names, colors)[0])
    
    plot_data_original, plot_data_interpo, intersections = plot_transformed_cdf_2(group_data, table_names, selected_groups, colors)
    encoded_plots.append(plot_data_original)
    encoded_plots.append(plot_data_interpo)

    # Initialize the ber_results list
    ber_results = []

    # Instead of just printing, also append each intersection to the ber_results list
    for intersection in intersections:
        print("intersections:")
        print(intersection)
        ber_results.append(intersection)

    # Generate tables for visualizing BER results
    encoded_sigma_image, encoded_ppm_image = plot_ber_tables(ber_results, table_names)
    encoded_plots.append(encoded_sigma_image)
    encoded_plots.append(encoded_ppm_image)

    #encoded_image1, encoded_image2 = plot_overlap_table(all_overlaps, table_names, selected_groups, data_matrix_size, num_of_groups)
    #encoded_plots.append(encoded_image1)
    #encoded_plots.append(encoded_image2)

    encoded_plot_for_avg = plot_average_values_table(avg_values, table_names, selected_groups)
    encoded_plots.append(encoded_plot_for_avg)

    encoded_plot_for_std = plot_std_values_table(std_values, table_names, selected_groups)
    encoded_plots.append(encoded_plot_for_std)

    # Generate and append a single combined window analysis table plot
    encoded_window_analysis_plot = plot_combined_window_analysis_table(aggregated_window_values)
    encoded_plots.append(encoded_window_analysis_plot)

    return encoded_plots
