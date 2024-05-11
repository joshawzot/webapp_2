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
    sigma_table.set_fontsize(10)
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
    ppm_table.set_fontsize(10)
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
        
        label = f'Combined Group {group_names[i]}' if group_names[i] not in added_to_legend else ""
        if label:
            added_to_legend.add(label)
        
        plt.plot(sorted_data, sigma_values, linestyle='-', linewidth=1, color=colors[i % len(colors)], label=label)
        plt.scatter(sorted_data, sigma_values, s=10, color=colors[i % len(colors)])

        transformed_data_groups.append((sorted_data, sigma_values))

    plt.xlabel('Transformed Data Value', fontsize=12)
    plt.ylabel('Sigma (Standard deviations)', fontsize=12)
    plt.title('Transformed CDF of Data by Groups')
    plt.legend()
    plt.grid(True)

    buf_transformed_cdf = BytesIO()
    plt.savefig(buf_transformed_cdf, format='png', bbox_inches='tight')
    plt.clf()  # Clear plot after saving
    buf_transformed_cdf.seek(0)
    plot_data_original = base64.b64encode(buf_transformed_cdf.getvalue()).decode('utf-8')
    buf_transformed_cdf.close()

    # Begin second plot for interpolation
    plt.figure(figsize=figsize)
    plt.xlim(global_x_min, global_x_max)

    intersections = []

    for i, (sorted_data, sigma_values) in enumerate(transformed_data_groups[:-1]):
        x1, y1 = sorted_data, -sigma_values
        x2, y2 = transformed_data_groups[i + 1][0], transformed_data_groups[i + 1][1]

        common_x_min = max(min(x1), min(x2))
        common_x_max = min(max(x1), max(x2))

        if common_x_min < common_x_max:  # Valid overlap exists
            common_x = np.linspace(common_x_min, common_x_max, num=1000)
            interp_func_y1 = interp1d(x1, y1)
            interp_func_y2 = interp1d(x2, y2)
            interp_y1 = interp_func_y1(common_x)
            interp_y2 = interp_func_y2(common_x)

            plt.plot(common_x, interp_y1, linestyle='-', color=colors[i], alpha=0.7)
            plt.plot(common_x, interp_y2, linestyle='-', color=colors[i + 1], alpha=0.7)

            idx_closest = np.argmin(np.abs(interp_y1 - interp_y2))
            intersections.append((common_x[idx_closest], interp_y1[idx_closest]))
            plt.scatter(common_x[idx_closest], interp_y1[idx_closest], color='red', s=50, zorder=5)

    plt.xlabel('Interpolated Data Value', fontsize=12)
    plt.ylabel('Sigma (Standard deviations)', fontsize=12)
    plt.title('Interpolated CDF Curves for BER Calculation')
    plt.grid(True)

    buf_interpolated_cdf = BytesIO()
    plt.savefig(buf_interpolated_cdf, format='png', bbox_inches='tight')
    plt.clf()  # Clear plot after saving
    buf_interpolated_cdf.seek(0)
    plot_data_interpo = base64.b64encode(buf_interpolated_cdf.getvalue()).decode('utf-8')
    buf_interpolated_cdf.close()

    return plot_data_original, plot_data_interpo, intersections

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
    table.set_fontsize(8)
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
    table.set_fontsize(8)
    table.scale(1, 1.2)

    plt.title('Standard Deviation Values Table')
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

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
    plot_data_original, plot_data_interpo, intersections = plot_transformed_cdf_3(all_groups, group_names, selected_groups, colors)
    # Adding plots to the encoded_plots list
    encoded_plots.append(plot_data_original)
    encoded_plots.append(plot_data_interpo)

    # Initialize the ber_results list
    ber_results = []

    # Instead of just printing, also append each intersection to the ber_results list
    for intersection in intersections:
        print("intersections:")
        print(intersection)
        ber_results.append(intersection)
    
    print(ber_results)
    print("AAA")
    # Generate tables for visualizing BER results
    encoded_sigma_image, encoded_ppm_image = plot_ber_tables_2(ber_results)
    encoded_plots.append(encoded_sigma_image)
    encoded_plots.append(encoded_ppm_image)

    return encoded_plots