import pandas as pd
from io import BytesIO
import base64
import matplotlib.pyplot as plt
from db_operations import create_connection, fetch_data, close_connection
import numpy as np
import matplotlib.cm as cm
import sys

def generate_new_data(data, x):
    new_data = []
    for i in range(len(data)):
        new_data.append(data[i])
        if i < len(data) - 1:
            new_start = data[i][1] + 1
            new_end = data[i+1][0] - 1
            if new_start <= new_end:
                new_data.append((new_start, new_end))
    
    # Add the last tuple with end value as x
    new_data.append((data[-1][1] + 1, x))
    return new_data

def generate_plot_ttc(table_name, database_name):
    # Connect to the MySQL database
    connection = create_connection(database_name)
    
    # Fetch the data directly into a pandas DataFrame
    query = f"SELECT * FROM `{table_name}`"
    df = pd.read_sql(query, connection)

    # Define the number of data sets and columns for each data set
    num_data_sets = (df.shape[1] + 1) // 4

    # Tolerance to consider a value as stable
    tolerance = 0.01

    plot_data_list = []

    # Loop through each data set and create separate graphs
    for i in range(num_data_sets):
        # Extract data for each data set
        graph_name = df.columns[i * 4] #header row
        x_axis_name = df.iat[0, i * 4]
        x_data_pre = df.iloc[1:, i * 4].astype(float)
        x_data = x_data_pre.dropna().astype(float)
        left_y_axis_name = df.iat[0, i * 4 + 1]
        left_y_data_pre = df.iloc[1:, i * 4 + 1].astype(float)
        left_y_data = left_y_data_pre.dropna().astype(float)
        right_y_axis_name = df.iat[0, i * 4 + 2]
        right_y_data_pre = df.iloc[1:, i * 4 + 2].astype(float)
        right_y_data = right_y_data_pre.dropna().astype(float)

        # Find and mark stable sections
        stable_sections = []
        current_section = []
        for j in range(len(x_data)):
            if not current_section or abs(right_y_data.iloc[j] - current_section[-1]) <= tolerance:
                current_section.append(right_y_data.iloc[j])
            else:
                if len(current_section) > 1:
                    stable_sections.append((x_data.index[j - len(current_section)], x_data.index[j - 1]))
                current_section = []
        
        all_sections = generate_new_data(stable_sections, len(x_data))

        # Plot the data for the left y-axis
        plt.figure()
        plt.plot(x_data, left_y_data, 'b-', label=left_y_axis_name)
        plt.ylabel(left_y_axis_name, color='b')
        plt.tick_params(axis='y', labelcolor='b')

        ax2 = plt.gca().twinx()
        ax2.plot(x_data, right_y_data, 'r-', label=right_y_axis_name)
        ax2.set_ylabel(right_y_axis_name, color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        for section_start, section_end in stable_sections:
            ax2.axvline(x=x_data.iloc[section_start], color='gray', linestyle='--')
            ax2.axvline(x=x_data.iloc[section_end], color='gray', linestyle='--')

        ax2.axvline(x=x_data.iloc[-1], color='gray', linestyle='--')

        plt.title(graph_name)
        plt.xlabel(x_axis_name)
        plt.legend(loc='best')

        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plot_data_list.append(base64.b64encode(buf.getvalue()).decode('utf-8'))
        buf.close()
        
        plt.figure()

        boxplot_data = []
        for section_start, section_end in all_sections:
            boxplot_data.append(left_y_data.iloc[section_start: section_end + 1].values)

        plt.boxplot(boxplot_data, showfliers=False)
        plt.xlabel('Section Number')
        plt.ylabel(left_y_axis_name)
        plt.title(f'Boxplot of {left_y_axis_name} in different sections')

        std_data = [data.std() for data in boxplot_data]

        ax3 = plt.gca().twinx()
        ax3.plot(range(1, len(all_sections) + 1), std_data, 'g--', label='Std Deviation')
        ax3.set_ylabel('Sigma', color='g')
        ax3.tick_params(axis='y', labelcolor='g')

        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plot_data_list.append(base64.b64encode(buf.getvalue()).decode('utf-8'))
        buf.close()
        plt.close('all')  # Close all open plots

    close_connection()
    
    return plot_data_list