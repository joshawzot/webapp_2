from tools_for_plots import *
import io
import base64
import pandas as pd
from config import MULTI_DATABASE_ANALYSIS
from scipy.stats import norm  # Import norm from scipy.stats

def plot_ber_tables_2(ber_results):
    def setup_figure():
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')
        return fig, ax
    
    # Headers for Sigma and PPM tables
    sigma_headers = ["Measurement", "Sigma Value"]
    ppm_headers = ["Measurement", "PPM"]

    sigma_data = [sigma_headers]
    ppm_data = [ppm_headers]

    # Populate table data
    for measurement, sigma in ber_results:
        ppm = sigma_to_ppm(abs(sigma))  # Convert sigma to ppm using the defined conversion function
        print(sigma)
        print(ppm)
        sigma_data.append([f"{measurement:.2f}", f"{sigma:.4f}"])
        ppm_data.append([f"{measurement:.2f}", f"{ppm:.0f}"])

    # Sigma Table
    fig_sigma, ax_sigma = setup_figure()
    sigma_table = ax_sigma.table(cellText=sigma_data, loc='center', cellLoc='center')
    sigma_table.auto_set_font_size(False)
    sigma_table.set_fontsize(12)
    sigma_table.scale(1, 1.5)
    plt.title("Sigma Values Table")

    buf_sigma = BytesIO()
    plt.savefig(buf_sigma, format='png', bbox_inches='tight')
    plt.close(fig_sigma)
    buf_sigma.seek(0)
    encoded_sigma_image = base64.b64encode(buf_sigma.getvalue()).decode('utf-8')

    # PPM Table
    fig_ppm, ax_ppm = setup_figure()
    ppm_table = ax_ppm.table(cellText=ppm_data, loc='center', cellLoc='center')
    ppm_table.auto_set_font_size(False)
    ppm_table.set_fontsize(12)
    ppm_table.scale(1, 1.5)
    plt.title("PPM Values Table")

    buf_ppm = BytesIO()
    plt.savefig(buf_ppm, format='png', bbox_inches='tight')
    plt.close(fig_ppm)
    buf_ppm.seek(0)
    encoded_ppm_image = base64.b64encode(buf_ppm.getvalue()).decode('utf-8')

    return encoded_sigma_image, encoded_ppm_image

def plot_transformed_cdf_3(data, group_names, selected_groups, colors, figsize=(15, 10)):
    plt.figure(figsize=figsize)
    added_to_legend = set()
    transformed_data_groups = []
    global_x_min = float('inf')
    global_x_max = float('-inf')
    ber_results = []

    for i, group_data in enumerate(data):
        if len(group_data) == 0:
            continue

        sorted_data = np.sort(group_data)
        global_x_min = min(global_x_min, sorted_data[0])
        global_x_max = max(global_x_max, sorted_data[-1])

        cdf_values = (np.arange(1, len(sorted_data) + 1) - 0.5) / len(sorted_data)
        sigma_values = norm.ppf(cdf_values)

        #label = f'Combined Group {group_names[i]}' if group_names[i] not in added_to_legend else ""
        #if label:
            #added_to_legend.add(label)
        
        #plt.plot(sorted_data, sigma_values, linestyle='-', linewidth=1, color=colors[i % len(colors)], label=label)
        plt.plot(sorted_data, sigma_values, linestyle='-', linewidth=1, color=colors[i % len(colors)])
        plt.scatter(sorted_data, sigma_values, s=10, color=colors[i % len(colors)])

        transformed_data_groups.append((sorted_data, sigma_values))

    plt.xlim(global_x_min, global_x_max)
    plt.xlabel('Transformed Data Value', fontsize=12)
    plt.ylabel('Sigma (Standard deviations)', fontsize=12)
    plt.title('Transformed CDF of Data by Groups')
    plt.legend()
    plt.grid(True)

    buf_transformed_cdf = BytesIO()
    plt.savefig(buf_transformed_cdf, format='png', bbox_inches='tight')
    buf_transformed_cdf.seek(0)
    plot_data_transformed_cdf = base64.b64encode(buf_transformed_cdf.getvalue()).decode('utf-8')
    buf_transformed_cdf.close()
    plt.clf()

    # Begin second plot for interpolation
    plt.figure(figsize=figsize)
    plt.xlim(global_x_min, global_x_max)
    intersections = []
    horizontal_line_y_value = []

    for i, (x1, y1) in enumerate(transformed_data_groups[:-1]):
        y1 = -y1
        x2, y2 = transformed_data_groups[i + 1]
        print("min(x1):", min(x1))
        print("min(x2):", min(x2))
        print("max(x1):", max(x1))
        print("max(x2):", max(x2))
        
        common_x_min = max(min(x1), min(x2))
        common_x_max = min(max(x1), max(x2))
        print("common_x_min:", common_x_min)
        print("common_x_max:", common_x_max)

        common_x_min_all = min(min(x1), min(x2))
        common_x_max_all = max(max(x1), max(x2))
        common_x_all = np.linspace(common_x_min_all, common_x_max_all, num=5000)

        common_x_1 = np.linspace(common_x_min_all, common_x_min, num=1000)
        common_x_2 = np.linspace(common_x_max, common_x_max_all, num=1000)

        '''remove the duplicated x for both curves'''
        # Create unique datasets by removing duplicates
        unique_x1, unique_indices_x1 = np.unique(x1, return_index=True)
        unique_y1 = y1[unique_indices_x1]

        unique_x2, unique_indices_x2 = np.unique(x2, return_index=True)
        unique_y2 = y2[unique_indices_x2]

        interp_common_x_1 = interp1d(unique_x1, unique_y1, fill_value="extrapolate")(common_x_all)
        interp_common_x_2 = interp1d(unique_x2, unique_y2, fill_value="extrapolate")(common_x_all)

        # Plotting the extrapolated parts with dashed lines
        plt.plot(common_x_all, interp_common_x_1, linestyle='-', color=colors[i], alpha=0.7)
        plt.plot(common_x_all, interp_common_x_2, linestyle='-', color=colors[i], alpha=0.7)

        idx_closest = np.argmin(np.abs(interp_common_x_1 - interp_common_x_2))
        intersection_x = common_x_all[idx_closest]
        intersection_y = interp_common_x_1[idx_closest]
        plt.scatter(intersection_x, intersection_y, color='red', s=50, zorder=5)
        intersections.append((intersection_x, intersection_y))

        # Existing calculation for the intersection
        print(f"Debug: Intersection at (x={intersection_x}, y={intersection_y})")

        # Calculate x-differences and plotting when they are about 2 units apart
        target_x_diff = 2
        tolerance = 0.1  # Smaller tolerance for precise calculation
        line_drawn = False  # Flag to ensure only one line is drawn

        for idx in range(len(common_x_all) - 1):
            for jdx in range(idx + 1, len(common_x_all)):
                x_diff = common_x_all[jdx] - common_x_all[idx]
                #print("x_diff:", x_diff)
                if abs(x_diff - target_x_diff) < tolerance:
                    #print(f"Debug: x_diff = {x_diff}, idx = {idx}, jdx = {jdx}")  # Debug output
                    if interp_common_x_2[jdx] > interp_common_x_1[idx]:  # Check divergence
                        if not line_drawn:
                            plt.hlines(y=interp_common_x_2[jdx], xmin=common_x_all[idx], xmax=common_x_all[jdx], color='green', linestyles='dotted')
                            print(f"Debug: Horizontal line drawn from x={common_x_all[idx]} to x={common_x_all[jdx]} at y={interp_common_x_2[jdx]}")
                            horizontal_line_y_value.append(interp_common_x_2[jdx])  # Store the y-value where the line is drawn
                            line_drawn = True
                            break
            if line_drawn:
                break

        if not line_drawn:
            print("No suitable points found to draw a horizontal line.")

        #plt.xlabel('Data Value', fontsize=12)
        plt.ylabel('Sigma (Standard deviations)', fontsize=12)
        plt.title('CDF Curves for BER Calculation')
        plt.grid(True)

    buf_interpolated_cdf = BytesIO()
    plt.savefig(buf_interpolated_cdf, format='png', bbox_inches='tight')
    plt.clf()  # Clear plot after saving
    buf_interpolated_cdf.seek(0)
    plot_data_interpo = base64.b64encode(buf_interpolated_cdf.getvalue()).decode('utf-8')
    buf_interpolated_cdf.close()

    #return plot_data_transformed_cdf, plot_data_interpo, intersections
    print("horizontal_line_y_value:", horizontal_line_y_value)
    return plot_data_transformed_cdf, plot_data_interpo, intersections, horizontal_line_y_value

def plot_boxplot_2(data, group_names, figsize=(15, 10)):
    plt.figure(figsize=figsize) 
    xticks = []
    xticklabels = group_names  # Labels for groups

    # Iterate over each group to plot
    for i, group_data in enumerate(data):
        start_position = i * len(data) + 1
        xticks.append(start_position)
        plt.boxplot(group_data, positions=[start_position], widths=0.6)

    plt.xticks(xticks, xticklabels, rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel('Value (unit)', fontsize=12)
    plt.grid(True)

    buf_boxplot = BytesIO()
    plt.savefig(buf_boxplot, format='png', bbox_inches='tight')
    buf_boxplot.seek(0)
    plot_data_boxplot = base64.b64encode(buf_boxplot.getvalue()).decode('utf-8')
    plt.clf()  # Clear the figure for fresh plots
    buf_boxplot.close()
    return plot_data_boxplot

def plot_average_values_table_2(avg_values, group_names, figsize=(12, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('tight')
    ax.axis('off')

    header = ["Group"] + ["Average"]
    table_data = [header]

    for i, avg in enumerate(avg_values):
        row = [group_names[i], f"{avg:.2f}"]
        table_data.append(row)

    column_widths = [0.2, 0.2]  # Adjust as needed

    table = ax.table(cellText=table_data, loc='center', colWidths=column_widths, cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.2)

    plt.title('Average Values Table')
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    encoded_image = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return encoded_image

def plot_std_values_table_2(std_values, group_names, figsize=(12, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('tight')
    ax.axis('off')

    header = ["Group"] + ["Standard Deviation"]
    table_data = [header]

    for i, std in enumerate(std_values):
        row = [group_names[i], f"{std:.2f}"]
        table_data.append(row)

    column_widths = [0.2, 0.2]  # Adjust as needed

    table = ax.table(cellText=table_data, loc='center', colWidths=column_widths, cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.2)

    plt.title('Standard Deviation Values Table')
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    encoded_image = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return encoded_image

def plot_2uS_table(ppm, selected_groups, figsize=(12, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('tight')
    ax.axis('off')

    # Specific pairs creation
    if len(selected_groups) >= 2:
        # Define specific pairs from selected_groups
        pairs = [(selected_groups[i], selected_groups[i + 1]) for i in range(len(selected_groups) - 1)]
        
        # Generate table data using the defined pairs and ppm values
        table_data = [[f"state{pair[0]} & state{pair[1]}", f"{ppm_val:.2f}"] for pair, ppm_val in zip(pairs, ppm)]
    else:
        # Fallback if not enough groups to form at least one pair
        table_data = []

    # Create the table
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.2)  # Adjust scale for better readability

    # Set the plot title and adjust the plot
    plt.title('2uS_ppm', fontsize=14, pad=15)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05)  # Adjust for title and better table fit

    # Save the plot to a buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    encoded_image = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return encoded_image

def generate_plot_normal_combined(table_names, database_name, form_data):
    print("table_names:", table_names)
    selected_groups = form_data.get('selected_groups', [])
    print("selected_groups:", selected_groups)
    
    sub_array_size_raw = form_data.get('sub_array_size', '324,64')  # Default to '324,64' if not present
    sub_array_size = tuple(sub_array_size_raw)
    print("sub_array_size:", sub_array_size)

    colors = get_colors(len(table_names))
    encoded_plots = []

    # Initialize structures for aggregated data
    aggregated_groups = {group: [] for group in selected_groups}
    aggregated_stats = {group: [] for group in selected_groups}

    # Process each table and collect data and statistics
    for table_name in table_names:
        groups, stats, _, num_of_groups, _ = get_group_data_new(table_name, selected_groups, database_name, sub_array_size)
        # Aggregate data and stats
        for group_index, group_data in enumerate(groups):
            group_key = selected_groups[group_index]  # Assuming groups correspond to selected_groups by index
            aggregated_groups[group_key].extend(group_data)  # Combine data for the same group across tables
            aggregated_stats[group_key].append(stats[group_index])  # Store stats for combining later

    # Now process the aggregated data
    all_groups = []
    avg_values = []
    std_values = []
    for group_key, group_data in aggregated_groups.items():
        all_groups.append(group_data)
        # Combine statistics: calculate overall average and standard deviation across tables
        all_avg = np.mean([stat[2] for stat in aggregated_stats[group_key]])  # Average of averages
        all_std = np.sqrt(sum([stat[3]**2 for stat in aggregated_stats[group_key]]) / len(aggregated_stats[group_key]))  # Pooled standard deviation
        avg_values.append(all_avg)
        std_values.append(all_std)

    group_names = [f'Group {i+1}' for i in range(len(all_groups))]
    encoded_plots = []

    # Call plot_boxplot_2 with correct variables
    plot_data_boxplot = plot_boxplot_2(all_groups, group_names)
    encoded_plots.append(plot_data_boxplot)

    encoded_plot_for_avg = plot_average_values_table_2(avg_values, group_names)
    encoded_plots.append(encoded_plot_for_avg)

    encoded_plot_for_std = plot_std_values_table_2(std_values, group_names)
    encoded_plots.append(encoded_plot_for_std)

    # If colors were specifically tailored for table_names, you may need to redefine colors:
    colors = get_colors(len(group_data))  # Assuming get_colors() generates a sufficient number of colors
    # Adjusted function call
    plot_data_original, plot_data_interpo, intersections, horizontal_line_y_value = plot_transformed_cdf_3(all_groups, group_names, selected_groups, colors)
    print("intersections:", intersections)
    print("horizontal_line_y_value:", horizontal_line_y_value)
    print("selected_groups:", selected_groups)

    tail_probability = sp_stats.norm.sf([abs(y) for y in horizontal_line_y_value])
    ppm = tail_probability * 1_000_000
    print("ppm:", ppm)
    print("A")
    table = plot_2uS_table(ppm, selected_groups)
    encoded_plots.append(table)
    # Adding plots to the encoded_plots list
    encoded_plots.append(plot_data_original)
    encoded_plots.append(plot_data_interpo)
    print("B")
    encoded_sigma_image, encoded_ppm_image = plot_ber_tables_2(intersections)
    encoded_plots.append(encoded_sigma_image)
    encoded_plots.append(encoded_ppm_image)
    print("C")
    normalized_groups, index_to_element = normalize_selected_groups(selected_groups)
    print("normalized_groups:", normalized_groups)
    print(f"Original: {selected_groups}, Normalized: {normalized_groups}")
    window_values_99, window_values_999 = calculate_window_values(all_groups, normalized_groups)
    print("D")
    #print("window_values:", window_values)
    # Generate and append a single combined window analysis table plot
    encoded_window_analysis_plot = plot_combined_window_analysis_table_2(window_values_99)
    encoded_plots.append(encoded_window_analysis_plot)
    encoded_window_analysis_plot = plot_combined_window_analysis_table_2(window_values_999)
    encoded_plots.append(encoded_window_analysis_plot)
    print("E")
    return encoded_plots