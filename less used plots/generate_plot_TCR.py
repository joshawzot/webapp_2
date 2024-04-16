import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
import base64
from db_operations import create_connection, close_connection
import re
#64x64x8

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

def create_plots_with_tcr(sorted_data, setting_index, levels_to_plot, connection, table_names):
    print("create_plots_with_tcr")
    encoded_plots = []  # Initialize list to store encoded plots
    plt.figure(figsize=(20, 10))  # Adjust the figure size as needed
    ax1 = plt.gca()  # Primary axis for boxplots
    ax2 = ax1.twinx()  # Secondary axis for TCR plots
    ax2.set_ylabel('TCR (1/C)')

    # Fetch reference data using the connection
    reference_table_name = next((name for name in table_names if "1_" in name), None)
    print('reference_table_name:', reference_table_name)

    # Fetch reference data using the connection
    reference_data = fetch_data_from_db(reference_table_name, connection)

    x_tick_labels = []  # List to store x-tick labels
    x_tick_positions = []  # List to store x-tick positions

    # Data structure to hold TCR values for each level
    tcr_values = {level: [] for level in levels_to_plot}

    # Variable to keep track of the x position for each set of boxplots
    current_position = 1

    for i, (filename, one_d_array) in enumerate(sorted_data.items()):
        # Append the filename to x_tick_labels
        x_tick_labels.append(filename)

        # Extract temperature using regex (case-insensitive)
        match = re.search(r'(\d+)[cC]', filename)
        if match:
            temp_str = match.group(1)
            temp_value = int(temp_str)
        else:
            print("Temperature value not found in filename:", filename)
            continue

        # Calculate the x positions for the current set of boxplots
        positions = list(range(current_position, current_position + len(levels_to_plot)))
        x_tick_positions.append(sum(positions) / len(positions))  # Center position for the label

        for level in levels_to_plot:
            # Extract level data and plot the boxplot
            start_index = setting_index * 256 * 16 + level * 256
            end_index = start_index + 256
            level_data = one_d_array[start_index:end_index] * 1e6
            ax1.boxplot(level_data, positions=[current_position], widths=0.6)

            # Calculate and store TCR values
            if temp_value != 25:
                ref_start_index = setting_index * 256 * 16 + level * 256
                ref_end_index = ref_start_index + 256
                g1 = np.mean(reference_data[ref_start_index:ref_end_index]) * 1e6
                g2 = np.mean(level_data)
                delta_t = temp_value - 25
                tcr_value = calculate_tcr(g1, g2, delta_t)
                tcr_values[level].append((current_position, tcr_value))
            
            current_position += 1  # Move to the next position for the next level

    # Plot TCR values
    for level, level_tcr_values in tcr_values.items():
        if level_tcr_values:  # If there are any TCR values to plot
            level_positions, tcrs = zip(*level_tcr_values)
            ax2.plot(level_positions, tcrs, label=f'Level {level}', marker='o')

    # Apply the labels and positions to the x-axis
    ax1.set_xticks(x_tick_positions)
    ax1.set_xticklabels(x_tick_labels, rotation=45, ha="right")

    # Set the y-axis label, plot title, and grid
    ax1.set_ylabel('Conductance (uS)')
    ax1.set_title(f'Setting Index: {setting_index}')
    ax1.grid(True)

    # Add legend for TCR
    ax2.legend(loc='upper left')

    # Tighten the layout
    plt.tight_layout()

    # Save the plot to a buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    # Encode plot data to base64
    encoded_plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    encoded_plots.append(encoded_plot_data)

    # Clean up
    buf.close()
    plt.close()

    return encoded_plots

def generate_plot_TCR(table_names, database_name):
    print("Table Names:", table_names)
    user_selected_levels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    connection = create_connection(database_name)
    table_names_sorted = sorted(table_names, key=sorting_key)
    print('table_names_sorted:', table_names_sorted)

    data = []
    for table_name in table_names_sorted:
        try:
            fetched_data = fetch_data_from_db(table_name, connection)
            data.append(fetched_data)
        except Exception as e:
            print(f"Error fetching data for {table_name}: {e}")
            continue

    sorted_data = {table_name: data[table_names_sorted.index(table_name)] for table_name in table_names_sorted}

    encoded_plots = []
    for setting_index in range(8):  # setting_index 0~7
        encoded_plots += create_plots_with_tcr(sorted_data, setting_index, user_selected_levels, connection, table_names)

    close_connection()
    return encoded_plots