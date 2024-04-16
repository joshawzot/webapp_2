import pandas as pd
from io import BytesIO
import base64
import matplotlib.pyplot as plt
from db_operations import create_connection, close_connection
import numpy as np
import re

def create_color_map(df, start_col, condition, cmap_name):
    condition_count = sum(condition in col_name for col_name in df.iloc[1, start_col+1:])
    return plt.get_cmap(cmap_name)(np.linspace(0.5, 1, condition_count))

def generate_plot_percentage_bars(table_name, database_name):
    connection = create_connection(database_name)
    
    df = pd.read_sql(f"SELECT * FROM `{table_name}`", connection)
    num_cols = df.shape[1]
    start_col = 0

    #print(df)
    # Step 1: Extract unique temperatures from column names using a regular expression and count their occurrences
    temp_counts = {}
    temp_pattern = re.compile(r'(\d+C)')

    column_names = df.columns.values  #taking the header row
    print('column_names:', column_names)
    
    for col_name in column_names[1:]:
        if pd.isnull(col_name):
            break
        matches = temp_pattern.findall(str(col_name))
        for match in matches:
            if match not in temp_counts:
                temp_counts[match] = 1
            else:
                temp_counts[match] += 1

    # Sort the unique temperatures and assign color maps
    unique_temps = sorted(temp_counts.keys(), key=lambda x: int(x[:-1]), reverse=True)  # Sorting temperatures in descending order
    print(unique_temps)

    # Step 2: Create a dictionary to map each unique temperature to a different color map with a number of shades equal to its count
    available_cmaps = ['Oranges', 'Greens', 'Blues']  # Extend this list with more color maps if necessary
    color_maps_and_indices = {temp: (plt.get_cmap(cmap)(np.linspace(0.5, 1, temp_counts[temp])), 0) for temp, cmap in zip(unique_temps, available_cmaps)}
    
    while start_col < num_cols:
        print('start_col:', start_col)
        fig, ax = plt.subplots(figsize=(18, 12))
        legend_entries = []

        for col_idx in range(start_col + 1, num_cols):
            col_name = df.columns[col_idx]
            print('col_name:', col_name)
            if pd.isnull(col_name):
                break

            x = pd.to_numeric(df.iloc[:, col_idx], errors='coerce')
            print('x:', x)
            width = 0.1
            column = df.iloc[:, col_idx]
            column_length = len(column.dropna())
            value_counts = x.value_counts(normalize=False, sort=False)
            percentages = value_counts / (column_length - 1) * 100

            start_points = (value_counts.index // width * width).values
            accumulated_percentages = [percentages[(start <= value_counts.index) & (value_counts.index < start + width)].sum() for start in start_points]

            for condition in unique_temps:  # Step 3: Dynamically set the color map based on the condition
                if condition in col_name:
                    color_map, color_index = color_maps_and_indices[condition]
                    color_index = color_index % len(color_map)  # Reset the color_index back to 0 if it exceeds the length of color_map
                    color = color_map[color_index]
                    ax.bar(start_points, accumulated_percentages, width=width, label=col_name, color=color)
                    color_maps_and_indices[condition] = (color_map, color_index + 1)
                    break
            else:
                color = None
                
            mean, sigma = x.mean(), x.std()
            legend_entries.append((plt.Rectangle((0, 0), 1, 1, fc=color, edgecolor='none'), f"{col_name} (Mean: {mean:.2f}, Sigma: {sigma:.2f})"))

        ax.legend(handles=[entry[0] for entry in legend_entries], labels=[entry[1] for entry in legend_entries], title='', fontsize=15, loc='upper left', bbox_to_anchor=(0, 1))
        plt.subplots_adjust(right=0.7)

        #ax.set_xlabel('Conductance', fontsize=20)
        #ax.set_ylabel('Percentage', fontsize=20)
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=6))  # Adjusting the number of x-axis ticks
        ax.yaxis.set_major_locator(plt.MultipleLocator(base=5))
        plt.xticks(rotation=45, fontsize=20)  # Rotating x-axis tick labels to avoid overlap
        plt.yticks(fontsize=20)

        start_col = col_idx + 1

    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    close_connection()

    return plot_data