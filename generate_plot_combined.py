import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from datetime import datetime
from io import BytesIO
import base64
import scipy.stats as sp_stats  # Alias for scipy.stats
from db_operations import create_connection, close_connection
import re
from scipy.stats import gamma
from scipy.integrate import quad

from fitter import Fitter
import scipy.stats as stats
from tools_for_plots import *

#combined
def plot_histogram_with_fitting(aggregated_data, colors, figsize=(15, 10)):
    encoded_plots_density = []
    encoded_plots_percentage = []
    fitting_params = {}

    # Since we're aggregating beforehand, we only have one "table" of aggregated groups
    table_name = "Aggregated Data"

    # Density plot setup
    fig_density, ax_density = plt.subplots(figsize=figsize)
    ax_density.set_title(f'Density and Best Fit for {table_name}')

    # Percentage plot setup
    fig_percentage, ax_percentage = plt.subplots(figsize=figsize)
    ax_percentage.set_title(f'Percentage for {table_name}')

    table_fitting_params = {}

    # Determine the overall range of the data
    all_data = np.hstack([subgroup[subgroup != 0] for subgroup in aggregated_data])  #filter out 0 in fitting
    #all_data = np.hstack(aggregated_data)
    data_range = np.max(all_data) - np.min(all_data)
    bin_width = data_range / 100  # For 100 bins
    bins = np.arange(np.min(all_data), np.max(all_data) + bin_width, bin_width)  # Bin edges

    for state_index, subgroup in enumerate(aggregated_data):
        subgroup = subgroup[subgroup != 0]   ## Exclude zeros from subgroup
        color = colors[state_index % len(colors)]

        # Density plot
        n, bins, patches = ax_density.hist(subgroup, bins=bins, color=color, alpha=0.6, density=True, label=f'State {state_index + 1} (Density)')

        # Percentage plot
        weights = np.ones_like(subgroup) / len(subgroup)
        ax_percentage.hist(subgroup, bins=bins, weights=weights, color=color, alpha=0.6, label=f'State {state_index + 1} (Percentage)')
        ax_percentage.set_yscale('log')

        # Fitting part
        f = Fitter(subgroup, bins=bins, distributions=['norm', 'expon', 'gamma', 'lognorm', 'beta'], timeout=30)
        f.fit()
        best_fit_name = list(f.get_best(method='sumsquare_error').keys())[0]
        best_fit_params = f.fitted_param[best_fit_name]

        table_fitting_params[state_index] = {'distribution': best_fit_name, 'parameters': best_fit_params}

        dist = getattr(stats, best_fit_name)
        lower_bound, upper_bound = dist.ppf([0.0000001, 0.9999999], *best_fit_params[:-2], loc=best_fit_params[-2], scale=best_fit_params[-1])
        x_range = np.linspace(lower_bound, upper_bound, 1000)
        pdf_values = dist.pdf(x_range, *best_fit_params[:-2], loc=best_fit_params[-2], scale=best_fit_params[-1])

        # Plot fitting for density
        ax_density.plot(x_range, pdf_values, linewidth=2, color=color, label=f'Best fit - State {state_index + 1}: {best_fit_name}')

        # Plot fitting for percentage
        percentage_fit_values = pdf_values * np.diff(bins)[0]
        ax_percentage.plot(x_range, percentage_fit_values, linewidth=2, color=color, label=f'Best fit - State {state_index + 1}: {best_fit_name}')

    # Finalize plots and encode
    for fig, buf_list in [(fig_density, encoded_plots_density), (fig_percentage, encoded_plots_percentage)]:
        buf = BytesIO()
        plt.tight_layout()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf_list.append(plot_data)
        buf.close()

    fitting_params[table_name] = table_fitting_params

    print("fitting_params:", fitting_params)
    return encoded_plots_density, encoded_plots_percentage, fitting_params

#combined
def plot_window_analysis_table(group_data, selected_groups, figsize=(12, 8)):
    """
    group_data is a dictionary where keys are group IDs and values are lists or NumPy arrays of data points for each group.
    """
    encoded_images = []
    window_analysis_data = {}

    # Iterate over the selected groups to calculate window values
    for i in range(len(selected_groups) - 1):
        state_a = selected_groups[i]
        state_b = selected_groups[i + 1]
        state_pair = f"State {state_a} & State {state_b}"

        # Fetch data for each state and ignore all zeros
        data_a = np.array(group_data[state_a])[np.array(group_data[state_a]) != 0]
        data_b = np.array(group_data[state_b])[np.array(group_data[state_b]) != 0]

        # Fetch data for each state
        #data_a = group_data[state_a]
        #data_b = group_data[state_b]

        # Calculate the 99th percentile of state_a and the 1st percentile of state_b
        percentile_99_a = np.percentile(data_a, 99)
        percentile_1_b = np.percentile(data_b, 1)

        # Calculate the window value difference
        print("percentile_1_b:", percentile_1_b)
        print("percentile_99_a:", percentile_99_a)
        window_value_difference =  percentile_1_b - percentile_99_a

        window_analysis_data[state_pair] = window_value_difference

    # Prepare table data
    header = ["State Pair", "Window Value"]
    table_data = [header] + [[pair, f"{vals:.2f}"] for pair, vals in window_analysis_data.items()]

    # Plot the window analysis table
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('tight')
    ax.axis('off')

    if len(table_data) > 1:
        table = ax.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.2)
        ax.set_title('Window Analysis for Selected Groups')
    else:
        ax.text(0.5, 0.5, "No window analysis data available", va='center', ha='center')

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    # Save and encode the plot image
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    encoded_image = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    encoded_images.append(encoded_image)

    return encoded_images

#combined
def generate_plot_combined(table_names, database_name, form_data):
    # Initialize list for all encoded plots
    encoded_plots = []

    # Prepare for data aggregation
    # Ensure this dictionary accumulates lists for concatenation
    aggregated_groups_by_selected_group = {group: [] for group in form_data.get('selected_groups', [])}
    selected_groups = form_data.get('selected_groups', [])

    # Fetch sub_array_size from form_data, which could be either a string or a list
    sub_array_size_raw = form_data.get('sub_array_size', '324,64')  # Default to '324,64' if not present
    print("sub_array_size_raw:", sub_array_size_raw)
    sub_array_size = tuple(sub_array_size_raw)

    # Collect and aggregate data from each table
    for table_name in table_names:
        groups, stats, _, _= get_group_data_new(table_name, selected_groups, database_name, sub_array_size)  #(row, column)
        #print(groups)
        for group_index, group_data in enumerate(groups):
            selected_group = selected_groups[group_index]
            print("selected_group:", selected_group)
            aggregated_groups_by_selected_group[selected_group].append(group_data)  # Append the group data as a list
            
    # Convert lists to NumPy arrays and then aggregate
    aggregated_groups = [np.concatenate(aggregated_groups_by_selected_group[group]) for group in selected_groups if len(aggregated_groups_by_selected_group[group]) > 0]

    # Set up color palette for plotting, adjusted for the number of selected groups
    cmap = cm.get_cmap('viridis')
    colors = [cmap(i / len(selected_groups)) for i in range(len(selected_groups))]

    # Process the aggregated dataset
    encoded_plots_density, encoded_plots_percentage, fitting_params = plot_histogram_with_fitting(
        aggregated_groups, colors
    )
    encoded_plots.extend(encoded_plots_density)
    encoded_plots.extend(encoded_plots_percentage)

    # Overlap calculation for the combined dataset
    specific_pairs = [(i, i + 1) for i in range(len(selected_groups) - 1)]
    overlap_ppm = calculate_overlap_fit(fitting_params["Aggregated Data"], specific_pairs)

    # Overlap plots
    combined_table_name = "Combined Data"
    overlap_ppm_by_table = {combined_table_name: overlap_ppm}
    encoded_combined_overlap_plot = plot_curve_overlap_table(overlap_ppm_by_table, {combined_table_name: selected_groups})
    #encoded_plots.extend([plot for _, plot in encoded_combined_overlap_plot])
    encoded_plots.append(encoded_combined_overlap_plot)

    # Statistics plot
    encoded_statistics_plot = plot_overlap_statistics(overlap_ppm_by_table, [combined_table_name])
    encoded_plots.append(encoded_statistics_plot)
    
    print("aggregated_groups:", aggregated_groups)
    # Instead of an aggregated_groups list, use a dictionary with selected_group IDs as keys.
    aggregated_groups_dict = {group: np.concatenate(aggregated_groups_by_selected_group[group]) for group in selected_groups if len(aggregated_groups_by_selected_group[group]) > 0}

    # Update the call to pass aggregated_groups_dict instead of aggregated_groups
    encoded_window_analysis_plot = plot_window_analysis_table(aggregated_groups_dict, selected_groups)

    encoded_plots.extend(encoded_window_analysis_plot)

    return encoded_plots



 




