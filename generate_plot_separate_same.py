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

#separate but on same plot
def generate_plot(table_names, database_name, form_data):
    # Initialize list for all encoded plots
    encoded_plots = []

    # Aggregate group data across all tables
    aggregated_groups = []
    aggregated_stats = []
    selected_groups = form_data.get('selected_groups', [])
    combined_table_name = "Combined Data"

    # Fetch sub_array_size from form_data, which could be either a string or a list
    sub_array_size_raw = form_data.get('sub_array_size', '324,64')  # Default to '324,64' if not present
    print("sub_array_size_raw:", sub_array_size_raw)
    sub_array_size = tuple(sub_array_size_raw)

    # Set up color palette for plotting
    cmap = cm.get_cmap('viridis')  # Example colormap, adjust as needed
    colors = [cmap(i / len(table_names)) for i in range(len(table_names))]

    # Collect data from each table
    for table_name in table_names:
        groups, stats, _, _ = get_group_data_new(table_name, selected_groups, database_name, sub_array_size)
        aggregated_groups.extend(groups)
        aggregated_stats.extend(stats)

    # Process the aggregated dataset
    encoded_plots_density, encoded_plots_percentage, fitting_params = plot_histogram_with_fitting(
        [aggregated_groups], [combined_table_name], colors
    )
    encoded_plots.extend(encoded_plots_density)
    encoded_plots.extend(encoded_plots_percentage)

    # Overlap calculation for the combined dataset
    specific_pairs = [(i, i + 1) for i in range(len(selected_groups) - 1)]
    overlap_ppm = calculate_overlap_fit(fitting_params[combined_table_name], specific_pairs)

    # Overlap plots
    overlap_ppm_by_table = {combined_table_name: overlap_ppm}
    encoded_combined_overlap_plot = plot_curve_overlap_table(overlap_ppm_by_table, {combined_table_name: selected_groups})
    encoded_plots.append(encoded_combined_overlap_plot)

    # Statistics plot
    encoded_statistics_plot = plot_overlap_statistics(overlap_ppm_by_table, [combined_table_name])
    encoded_plots.append(encoded_statistics_plot)

    return encoded_plots



 




