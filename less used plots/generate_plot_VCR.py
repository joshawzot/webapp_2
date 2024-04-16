import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from datetime import datetime
from io import BytesIO
import base64
from db_operations import create_connection, close_connection
import re
#64x64x8

'''def extract_temp_info(temp_part):
    """Extract temperature value and round information from filename."""
    temp_value = int(temp_part.replace('c', '').split('_')[0])
    is_second_round = '2ndround' in temp_part
    return temp_value, is_second_round'''

def extract_temp_info(temp_part):
    """Extract temperature value and round information from filename."""
    temp_value = int(''.join(filter(str.isdigit, temp_part.split('_')[1])))  # Extract temperature value
    round_part = temp_part.split('_')[-1].lower()
    round_mapping = {'1stround': 1, '2ndround': 2, '3stround': 3, '4stround': 4}
    is_round = round_mapping.get(round_part, 1)  # Default to 1st round if not specified
    return temp_value, is_round

def sorting_key(table_name):
    try:
        # Split the table name at the first underscore and extract the first part
        first_part = table_name.split('_', 1)[0]

        # Find the sequence of digits in the first part
        match = re.search(r'\d+', first_part)

        if match:
            # Convert the extracted number to an integer
            return int(match.group())
        else:
            # If no number is found, return a high value to sort it last
            return float('inf')
    except ValueError as e:
        print(f"Error processing {table_name}: {e}")
        # In case of any error, return a high value to sort it last
        return float('inf')

def fetch_data(cursor, query):
    """Fetch data from the database using the provided cursor and query."""
    cursor.execute(query)
    return cursor.fetchall()

def process_data(data):
    """Process the fetched data for plotting."""
    # Assuming data needs to be converted to a numpy array and flattened
    return np.array(data).flatten()

def sort_table_names(table_names):
    """Sort table names based on custom criteria."""
    return sorted(table_names, key=sorting_key)

def plot_data(table_names_sorted, all_data, user_selected_levels, selected_setting, num_settings):
    """Generate and save boxplots and statistics tables as separate images."""
    encoded_boxplots = []
    encoded_tables = []
    for table_name, one_d_array in zip(table_names_sorted, all_data):
        boxplot, table = create_boxplot(table_name, one_d_array, user_selected_levels, selected_setting, num_settings)
        encoded_boxplots.append(boxplot)
        encoded_tables.append(table)
    return encoded_boxplots, encoded_tables

def create_boxplot(table_name, one_d_array, user_selected_levels, selected_settings, num_settings):
    """Create a boxplot and a statistics table for a single table, as separate images."""
    num_settings, start_voltage, end_voltage = 4, 0.05, 0.125
    voltage_step = (end_voltage - start_voltage) / (num_settings - 1)
    voltage_labels = [f'{start_voltage + i * voltage_step:.3f}V' for i in range(num_settings)]

    statistics = []

    # Create and save the boxplot figure
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    for setting_index in selected_settings:
        plot_setting_data(setting_index, one_d_array, user_selected_levels, voltage_labels, statistics, ax=ax1)

    num_levels = len(user_selected_levels)
    tick_positions = [(index * num_levels + 1) for index in selected_settings]
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels([voltage_labels[index] for index in selected_settings])
    ax1.set_ylabel('Conductance (uS)')
    ax1.set_title(f'Data from {table_name}')
    ax1.grid(True)

    buffer1 = BytesIO()
    plt.savefig(buffer1, format='png')
    plt.close(fig1)
    buffer1.seek(0)
    boxplot_data = base64.b64encode(buffer1.read()).decode('utf-8')

    # Adjust the size of the statistics table figure
    fig2, ax2 = plt.subplots(figsize=(12, 4 + len(statistics) * 0.3))  # Adjust the height based on the number of rows in the table
    ax2.axis('tight')
    ax2.axis('off')
    table = ax2.table(cellText=statistics, colLabels=["Setting Index", "Level", "Average (uS)", "Std Dev (uS)", "Outlier %"], loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)

    plt.subplots_adjust(left=0.2, bottom=0.2, right=0.8, top=0.8)  # Adjust these parameters as needed
    buffer2 = BytesIO()
    plt.savefig(buffer2, format='png', bbox_inches='tight')  # bbox_inches='tight' might help in fitting the table
    plt.close(fig2)
    buffer2.seek(0)
    table_data = base64.b64encode(buffer2.read()).decode('utf-8')

    return boxplot_data, table_data

def calculate_statistics(level_data):
    """Calculate statistics for a given set of level data and round to two decimal places."""
    average = round(np.mean(level_data), 2)
    std_dev = round(np.std(level_data), 2)
    outlier_percentage = round(np.sum(np.abs(level_data - average) > 2.698 * std_dev) / len(level_data) * 100, 2) # A value is considered an outlier if it is more than two standard deviations away from
    return average, std_dev, outlier_percentage

'''def plot_setting_data(setting_index, one_d_array, user_selected_levels, voltage_labels, statistics, ax):
    """Plot data for a specific setting and calculate statistics, using voltage labels as setting index."""
    for level in user_selected_levels:
        start_index = setting_index * 256 * 16 + level * 256
        level_data = one_d_array[start_index:start_index + 256] * 1e6
        ax.boxplot(level_data, positions=[setting_index * len(user_selected_levels) + user_selected_levels.index(level) + 1], widths=0.6)
        # Calculate and store statistics
        average, std_dev, outlier_percentage = calculate_statistics(level_data)
        setting_label = voltage_labels[setting_index]  # Use voltage label as setting index
        statistics.append((setting_label, level, average, std_dev, outlier_percentage))'''

def plot_setting_data(setting_index, one_d_array, user_selected_levels, voltage_labels, statistics, ax):
    """Plot data for a specific setting and calculate statistics, using voltage labels as setting index."""
    for level in user_selected_levels:
        start_index = setting_index * 256 * 16 + level * 256
        level_data = one_d_array[start_index:start_index + 256] * 1e6
        box = ax.boxplot(level_data, positions=[setting_index * len(user_selected_levels) + user_selected_levels.index(level) + 1], widths=0.6)
        # Calculate and store statistics
        average, std_dev, outlier_percentage = calculate_statistics(level_data)
        setting_label = voltage_labels[setting_index]  # Use voltage label as setting index
        statistics.append((setting_label, level, average, std_dev, outlier_percentage))

        # Annotating the boxplot with the calculated statistics
        x_position = setting_index * len(user_selected_levels) + user_selected_levels.index(level) + 1
        ax.text(x_position, max(level_data), f'Avg: {average}\nSD: {std_dev}\nOutliers: {outlier_percentage}%', horizontalalignment='center', size=8, color='black', weight='semibold')

def clean_and_convert(value):
    # Check if the value is directly an integer or a float
    if isinstance(value, (int, float)):
        return int(value)
    elif isinstance(value, str):
        # If the value is a string, attempt to strip and convert
        try:
            return int(float(value.strip()))
        except ValueError:
            return 0  # Default value or raise an error
    elif isinstance(value, list) and value:
        # If the value is a list, recursively call this function on its first item
        return clean_and_convert(value[0])
    else:
        return 0  # Default value or raise an error for unsupported types or empty list

def clean_and_convert_list(value):
    # Initialize an empty list to hold converted values
    converted_values = []

    if isinstance(value, str):
        # Process string to convert to list
        cleaned_values = value.strip("[] ").split(',')
    elif isinstance(value, (list, tuple)):
        # Use directly if it's already a list or tuple
        cleaned_values = value
    else:
        # Return an empty list for unsupported types or raise an error
        return []

    # Iterate over items, converting each to an integer
    for num in cleaned_values:
        try:
            # Directly convert integers or floats, strip and convert strings
            if isinstance(num, (int, float)):
                converted_values.append(int(num))
            elif isinstance(num, str):
                converted_values.append(int(float(num.strip())))
        except ValueError:
            # Skip items that can't be converted
            continue

    return converted_values

def generate_plot_VCR(table_names, database_name, form_data):
    all_data = []
    print("Table Names:", table_names)
    print("database_name:", database_name)

    connection = create_connection(database_name)
    cursor = connection.cursor()

    print("over")

    for table_name in table_names:
        print("table_name", table_name)
        query = f"SELECT * FROM {table_name}"
        cursor.execute(query)
        data = cursor.fetchall()

        processed_data = process_data(data)
        all_data.append(processed_data)

    print("over2")

    user_selected_levels = clean_and_convert_list(form_data.get('selected_groups', ''))
    print("user_selected_levels:", user_selected_levels)

    selected_setting = clean_and_convert_list(form_data.get('selected_setting', ''))
    num_settings = clean_and_convert(form_data.get('num_settings', '8'))
    print("selected_setting", selected_setting)
    print("num_settings:", num_settings)

    table_names_sorted = sort_table_names(table_names)
    print('Sorted Table Names:', table_names_sorted)

    #user_selected_levels = [0, 5, 10, 15]
    #user_selected_levels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    #user_selected_levels = [0]
    # Generate encoded boxplots and tables
    encoded_boxplots, encoded_tables = plot_data(table_names_sorted, all_data, user_selected_levels, selected_setting, num_settings)

    close_connection()

    # Interleave boxplots and tables in the final list
    encoded_plots = []
    for boxplot, table in zip(encoded_boxplots, encoded_tables):
        encoded_plots.append(boxplot)
        encoded_plots.append(table)

    return encoded_plots
