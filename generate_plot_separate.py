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

#separate
def plot_histogram_with_fitting(data, table_names, colors, figsize=(15, 10)):
    encoded_plots_density = []  # Density plots data
    encoded_plots_counts = []  # Counts plots data
    encoded_plots_percentage = []  # Percentage plots data
    fitting_params = {}  # Fitting parameters by table

    for table_index, table_data in enumerate(data):
        # Density plot setup
        fig_density, ax_density = plt.subplots(figsize=figsize)
        ax_density.set_title(f'Density and Best Fit for {table_names[table_index]}')

        # Counts plot setup
        fig_counts, ax_counts = plt.subplots(figsize=figsize)
        ax_counts.set_title(f'Counts with Log Scale for {table_names[table_index]}')

        # Percentage plot setup
        fig_percentage, ax_percentage = plt.subplots(figsize=figsize)
        ax_percentage.set_title(f'Percentage for {table_names[table_index]}')

        table_fitting_params = {}  # Fitting parameters for the current table

        for state_index, subgroup in enumerate(table_data):
            color = colors[state_index % len(colors)]

            # Density plot
            n, bins, patches = ax_density.hist(subgroup, bins=50, color=color, alpha=0.6, density=True, label=f'State {state_index + 1} (Density)')

            # Counts plot
            counts, bins, patches = ax_counts.hist(subgroup, bins=bins, color=color, alpha=0.6, log=True, label=f'State {state_index + 1} (Counts)')
            total = sum(counts)  # Define total here after counts plot

            # Percentage plot
            weights = np.ones_like(subgroup) / len(subgroup)
            ax_percentage.hist(subgroup, bins=bins, weights=weights, color=color, alpha=0.6, label=f'State {state_index + 1} (Percentage)')
            ax_percentage.set_yscale('log')

            # Fitting part
            f = Fitter(subgroup, distributions=['norm', 'expon', 'gamma', 'lognorm', 'beta'], timeout=30)
            f.fit()
            best_fit_name = list(f.get_best(method='sumsquare_error').keys())[0]
            best_fit_params = f.fitted_param[best_fit_name]

            # Store the fitting parameters for each state in the table_fitting_params dictionary
            table_fitting_params[state_index] = {'distribution': best_fit_name, 'parameters': best_fit_params}

            dist = getattr(stats, best_fit_name)
            lower_bound, upper_bound = dist.ppf([0.0000001, 0.9999999], *best_fit_params[:-2], loc=best_fit_params[-2], scale=best_fit_params[-1])
            x_range = np.linspace(lower_bound, upper_bound, 1000)
            pdf_values = dist.pdf(x_range, *best_fit_params[:-2], loc=best_fit_params[-2], scale=best_fit_params[-1])

            # Plot fitting for density
            ax_density.plot(x_range, pdf_values, linewidth=2, color=color, label=f'Best fit - State {state_index + 1}: {best_fit_name}')

            # Plot fitting for counts
            count_fit_values = pdf_values * np.diff(bins)[0] * total
            ax_counts.plot(x_range, count_fit_values, linewidth=2, color=color, label=f'Best fit - State {state_index + 1}: {best_fit_name}')

            # Plot fitting for percentage
            percentage_fit_values = pdf_values * np.diff(bins)[0]
            ax_percentage.plot(x_range, percentage_fit_values, linewidth=2, color=color, label=f'Best fit - State {state_index + 1}: {best_fit_name}')

        # Finalize density plot
        ax_density.set_xlabel('Value')
        ax_density.set_ylabel('Density')
        #ax_density.legend()

        # Finalize counts plot
        ax_counts.set_xlabel('Value')
        ax_counts.set_ylabel('Count')
        #ax_counts.legend()
        #ax_counts.set_ylim(bottom=0.9)

        # Finalize percentage plot
        ax_percentage.set_xlabel('Value')
        ax_percentage.set_ylabel('Percentage')
        #ax_percentage.legend()

        # Saving plots to buffer and encoding
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

    #return encoded_plots_density, encoded_plots_counts, encoded_plots_percentage, fitting_params
    print("fitting_params:", fitting_params)
    return encoded_plots_density, encoded_plots_percentage, fitting_params

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

def plot_combined_window_analysis_table(aggregated_window_values, selected_groups, figsize=(12, 8)):
    tables_order = sorted(set(table_name for (table_name, _), _ in aggregated_window_values.items()))

    # Initialize an empty list to maintain the order of state pairs based on their appearance
    state_pairs_order = []
    for _, pair in aggregated_window_values.keys():
        state_pair = f"Group {pair[0]} & Group {pair[1]}"
        if state_pair not in state_pairs_order:
            state_pairs_order.append(state_pair)

    # Initialize combined data with zeros for all state pairs across all tables
    combined_table_data = {state_pair: [0] * len(tables_order) for state_pair in state_pairs_order}

    # Populate the combined table data with actual values
    for (table_name, pair), value in aggregated_window_values.items():
        state_pair = f"Group {pair[0]} & Group {pair[1]}"
        table_index = tables_order.index(table_name)
        combined_table_data[state_pair][table_index] = value
    
    # Prepare table data with headers and rows
    header = ["State Pair"] + tables_order
    # Use the ordered list of state pairs to maintain the desired order in the final table
    table_data = [header] + [[state_pair] + [f"{vals[i]:.2f}" for i in range(len(vals))] for state_pair, vals in combined_table_data.items() if state_pair in state_pairs_order]

    # Plot combined table
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('tight')
    ax.axis('off')

    if len(table_data) > 1:  # Ensure there's data beyond the header
        table = ax.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.2)
        ax.set_title('Combined Window Analysis Across Tables', fontweight="bold")
    else:
        ax.text(0.5, 0.5, "No window analysis data available", va='center', ha='center')

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    encoded_image = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    return encoded_image

def normalize_selected_groups(selected_groups):
    # Create a sorted set of the unique elements in selected_groups
    unique_sorted_elements = sorted(set(selected_groups))
    
    # Create a dictionary that maps each unique element to its index in the sorted list
    element_to_index = {element: index for index, element in enumerate(unique_sorted_elements)}
    
    # Map each element in selected_groups to its new value using the dictionary
    normalized_groups = [element_to_index[element] for element in selected_groups]
    
    return normalized_groups

#separate
def generate_plot_separate(table_names, database_name, form_data):
    connection = create_connection(database_name)
    selected_groups = form_data.get('selected_groups', [])

    # Fetch sub_array_size from form_data, which could be either a string or a list
    sub_array_size_raw = form_data.get('sub_array_size', '324,64')  # Default to '324,64' if not present
    print("sub_array_size_raw:", sub_array_size_raw)
    sub_array_size = tuple(sub_array_size_raw)

    #specific_pairs = [(i, i + 1) for i in range(len(selected_groups) - 1)]
    #print("specific_pairs:", specific_pairs)

    normalized_groups = normalize_selected_groups(selected_groups)
    print(f"Original: {selected_groups}, Normalized: {normalized_groups}")

    encoded_plots = []
    fitting_params_by_table = {}
    aggregated_window_values = {}  # Aggregated window values across tables
    cmap = cm.get_cmap('viridis', len(table_names))
    norm = mcolors.Normalize(vmin=0, vmax=len(table_names) - 1)
    colors = [cmap(norm(i)) for i in range(len(table_names))]
    overlap_ppm_by_table = {}

    for table_name in table_names:
        groups, stats, _, _= get_group_data_new(table_name, selected_groups, database_name, sub_array_size)
        encoded_plot1, encoded_plot2, table_fitting_params = plot_histogram_with_fitting([groups], [table_name], colors)
        fitting_params_by_table[table_name] = table_fitting_params[table_name]
        encoded_plots.extend(encoded_plot1)
        encoded_plots.extend(encoded_plot2)

        # Calculate window values for this table and aggregate them
        window_values = calculate_window_values(groups, normalized_groups)
        for pair, value in window_values.items():
            aggregated_window_values[(table_name, pair)] = value

    # Generate and append a single combined window analysis table plot
    encoded_window_analysis_plot = plot_combined_window_analysis_table(aggregated_window_values, selected_groups)
    encoded_plots.append(encoded_window_analysis_plot)

    # Generate specific desired pairs from selected_groups for overlap calculations
    specific_pairs = [(i, i + 1) for i in range(len(selected_groups) - 1)]

    for table_name in table_names:
        table_fitting_params = fitting_params_by_table[table_name]
        overlap_ppm = calculate_overlap_fit(table_fitting_params, specific_pairs)
        filtered_overlap_ppm = {pair: overlap_ppm.get(pair, 0) for pair in specific_pairs}
        overlap_ppm_by_table[table_name] = filtered_overlap_ppm

    # Generate combined overlap plot and statistics plot
    encoded_combined_overlap_plot = plot_curve_overlap_table(overlap_ppm_by_table, {table_name: selected_groups for table_name in table_names})
    encoded_plots.append(encoded_combined_overlap_plot)
    encoded_statistics_plot = plot_overlap_statistics(overlap_ppm_by_table, table_names)
    encoded_plots.append(encoded_statistics_plot)

    return encoded_plots



 




