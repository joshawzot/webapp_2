import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
import base64
from db_operations import create_connection, close_connection
import re
#64x64x8

def generate_temp_labels_with_postfix(table_names):
    # This dictionary will store the last index used for each temperature
    last_index_used = {'25C': 0, '55C': 0, '85C': 0}
    # This list will store the final temperature labels with postfix
    temp_labels_with_postfix = []

    for name in table_names:
        # Extract the temperature from the table name using regex
        temp_match = re.search(r'(\d+C)', name)
        if temp_match:
            temp = temp_match.group(1)
            # For '25C', always increment the index
            if temp == '25C':
                last_index_used[temp] += 1
            # For '55C', increment the index only if the last '25C' has a higher index
            elif temp == '55C' and last_index_used['25C'] > last_index_used['55C']:
                last_index_used[temp] += 1
            # For '85C', increment the index only if the last '55C' has a higher index
            elif temp == '85C' and last_index_used['55C'] > last_index_used['85C']:
                last_index_used[temp] += 1

            # Use the last index used for this temperature as the postfix
            temp_labels_with_postfix.append(f"{temp}_{last_index_used[temp]}")
        else:
            temp_labels_with_postfix.append('Unknown')

    return temp_labels_with_postfix
    
def fetch_data_from_db(table_name, connection):
    table_name = f"`{table_name}`"
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, connection)
    return df.to_numpy()

def calculate_tcr(g1, g2, delta_t):
    return (g1 - g2) / g2 / delta_t

def flatten_sections(data_array, setting_index):
    # Check the dimension of the data_array
    print(f"Data array dimensions: {data_array.shape}")

    if len(data_array.shape) == 2:
        # Handle 2D data_array if necessary
        # Adjust this section based on how you want to process 2D data
        flattened_sections = [data_array.flatten()]
    elif len(data_array.shape) == 3:
        flattened_sections = []
        for row in range(0, 64, 16):
            for col in range(0, 64, 16):
                section = data_array[row:row+16, col:col+16, setting_index]
                flattened_sections.append(section.flatten())
    else:
        raise ValueError(f"Unexpected number of dimensions: {len(data_array.shape)}")
    return np.concatenate(flattened_sections)

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

def calculate_statistics(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    outlier_percentage = round(np.sum(np.abs(data - mean) > 2.698 * std_dev) / len(data) * 100, 2) # A value is considered an outlier if it is more than two standard deviations away from
    return mean, std_dev, outlier_percentage

def create_plots_with_tcr(sorted_data, setting_index, levels_to_plot, table_names, connection):
    print("create_plots_with_tcr")
    encoded_plots = []  # Initialize list to store encoded plots
    tcr_values = {level: [] for level in levels_to_plot}
    print('table_names:', table_names)

    # Fetch reference data using the connection
    reference_table_name = next((name for name in table_names if "1_" in name), None)
    print('reference_table_name:', reference_table_name)
    reference_data = fetch_data_from_db(reference_table_name, connection)

    x_labels = []
    all_level_statistics = []

    # Extract temperature values from table names for use as x-tick labels
    temp_labels = [re.search(r'\d+C', name).group() if re.search(r'\d+C', name) else 'Unknown' for name in table_names]

    plt.figure(figsize=(12, 8))
    ax1 = plt.gca()
    ax1.set_ylabel('Conductance (uS)')
    ax1.set_title(f'Boxplots for Setting Index: {setting_index}')
    ax1.grid(True)

    temp_label_positions = []  # To track positions of labels on the x-axis

    for i, (filename, one_d_array) in enumerate(sorted_data.items()):
        temp_value = None  # Initialize temp_value to None or a default value

        # Extract the temperature value (case-insensitive) and use as x-label
        match = re.search(r'(\d+)[cC]', filename)
        if match:
            temp_str = match.group(1)
            temp_value = int(temp_str)  # Update temp_value here
            print('temp_value:', temp_value)
        else:
            print("Temperature value not found in filename:", filename)
            continue  # Skip the rest of the loop if temp_value is not found

        for level in levels_to_plot:
            start_index = setting_index * 256 * 16 + level * 256
            end_index = start_index + 256
            level_data = one_d_array[start_index:end_index] * 1e6

            positions = [i * len(levels_to_plot) + levels_to_plot.index(level) + 1]
            box = ax1.boxplot(level_data, positions=positions, widths=0.6)

            # Calculate statistics
            mean, std_dev, outlier_percentage = calculate_statistics(level_data)
            all_level_statistics.append([level, round(mean, 2), round(std_dev, 2), round(outlier_percentage, 2)])

            if temp_value != 25:
                ref_start_index = setting_index * 256 * 16 + level * 256
                ref_end_index = ref_start_index + 256
                g1 = np.mean(reference_data[ref_start_index:ref_end_index]) * 1e6
                g2 = np.mean(level_data)
                delta_t = temp_value - 25
                tcr_value = calculate_tcr(g1, g2, delta_t)
                tcr_values[level].append((i, tcr_value))

        temp_label_positions.append(positions[0])  # Track position for temperature label

    # Inside your plotting function, after sorting or preparing table_names
    temp_labels = generate_temp_labels_with_postfix(table_names)

    # Adjust x-tick labels to show temperature values instead of full filenames
    ax1.set_xticks(temp_label_positions)
    ax1.set_xticklabels(temp_labels, rotation=45, ha="right")

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    encoded_plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    encoded_plots.append(encoded_plot_data)
    buf.close()
    plt.close()

    # Before plotting, extract temperature values from table names
    temp_labels = []
    for name in table_names:
        # Use regex to find temperature values in the format XXC within the table name
        match = re.search(r'\d+C', name)
        if match:
            temp_labels.append(match.group())
        else:
            temp_labels.append('Unknown')  # Fallback label if no temperature found

    temp_labels = generate_temp_labels_with_postfix(table_names)

    plt.figure(figsize=(12, 8))
    ax2 = plt.gca()
    ax2.set_xlabel('Table Name')
    ax2.set_xticks(range(len(temp_labels)))  # Set x-ticks based on the number of temperature labels
    ax2.set_xticklabels(temp_labels, rotation=45, ha="right")  # Use extracted temperature labels
    ax2.set_ylabel('TCR (1/C)')
    ax2.set_title(f'TCR Curves for Setting Index {setting_index}')
    ax2.grid(True)

    for level, level_tcr_values in tcr_values.items():
        if level_tcr_values:
            level_positions, tcrs = zip(*level_tcr_values)
            ax2.plot(level_positions, tcrs, label=f'Level {level}', marker='o', linestyle='-')
    ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))

    tcr_buf = BytesIO()
    plt.savefig(tcr_buf, format='png')
    tcr_buf.seek(0)
    encoded_tcr_plot = base64.b64encode(tcr_buf.getvalue()).decode('utf-8')
    encoded_plots.append(encoded_tcr_plot)
    tcr_buf.close()
    plt.close()

    # Increase the width of the figure if there are many columns or the height if there are many rows
    fig_width = max(15, len(levels_to_plot) * 2)  # Adjust width
    fig_height = max(10, len(levels_to_plot) * 1)  # Adjust height
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    ax.axis('off')
    ax.axis('tight')

    col_labels = ["Level", "Mean", "Std Dev", "% Outliers"]
    table = ax.table(cellText=all_level_statistics, colLabels=col_labels, loc='center', cellLoc='center')

    # Adjusting the font size and scaling the table
    table.set_fontsize(15)  # Adjust the font size as needed
    table.scale(1, 2)       # Scale the table dimensions; adjust as needed

    # Save the statistics table to a buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)  # Saving with high resolution
    buf.seek(0)
    encoded_plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    encoded_plots.append(encoded_plot_data)
    buf.close()
    plt.close()

    return encoded_plots  # Return the list of all encoded plot data

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

def generate_plot_TCR_separate(table_names, database_name, form_data):
    print("Table Names:", table_names)
    #user_selected_levels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    connection = create_connection(database_name)
    table_names_sorted = sorted(table_names, key=sorting_key)
    print('table_names_sorted:', table_names_sorted)

    data = []
    for table_name in table_names_sorted:
        try:
            fetched_data = fetch_data_from_db(table_name, connection)
            data_size = fetched_data.shape  # Get the dimensions of fetched_data
            print(f"fetched_data size for {table_name}: {data_size}")  # This line prints the size
            data.append(fetched_data)
        except Exception as e:
            print(f"Error fetching data for {table_name}: {e}")
            continue

    user_selected_levels = clean_and_convert_list(form_data.get('selected_groups', ''))
    print("user_selected_levels:", user_selected_levels)

    num_settings = clean_and_convert(form_data.get('num_settings'))
    
    sorted_data = {table_name: data[table_names_sorted.index(table_name)] for table_name in table_names_sorted}
    
    print("num_settings", num_settings)
    encoded_plots = []
    for setting_index in range(num_settings):  # setting_index 0~7
        encoded_plots += create_plots_with_tcr(sorted_data, setting_index, user_selected_levels, table_names_sorted, connection)

    close_connection()
    return encoded_plots
