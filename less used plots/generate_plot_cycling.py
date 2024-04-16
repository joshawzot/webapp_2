import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from io import BytesIO
import base64
import re
from db_operations import create_connection, close_connection

#64x64
def generate_plot_cycling(table_names, database_name):
    print(f"Database Name: {database_name}")  # Debugging line
    # Updated pattern to match "2023_12_19_202159_g_read_all_mat"
    #pattern = re.compile(r'\d{4}_\d{2}_\d{2}_\d{6}_g_read_all_mat')
    #pattern = re.compile(r'\d{4}_\d{2}_\d{2}_\d{6}')
    pattern = re.compile(r'G_read_\d+\_mat_cycling')
    # Sort the table names based on the pattern
    filtered_sorted_table_names = sorted(
        [t for t in table_names if pattern.match(t)],
        key=lambda x: x
    )

    # If you want to see what's being matched
    print(filtered_sorted_table_names)
    print('_____________________')

    plt.figure(figsize=(20, 10))
    colors = plt.cm.tab20(np.linspace(0, 1, 20))

    for index, table_name in enumerate(filtered_sorted_table_names):
        print(f"Database Name: {database_name}") 
        connection = create_connection(database_name)
        query = f"SELECT * FROM {table_name}"  # Adjust this query based on your needs

        # Execute the query and fetch data
        cursor = connection.cursor()
        cursor.execute(query)
        data = cursor.fetchall()  # or use fetchone(), fetchmany() based on how you want to retrieve the data

        data_array = np.array(data)  # Convert the list of tuples to a NumPy array
        data_flattened = data_array.flatten()

        # Calculate the empirical CDF
        sorted_data = np.sort(data_flattened) * 1e6
        #cdf_values = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        cdf_values = (np.arange(1, len(sorted_data) + 1) - 0.5) / len(sorted_data)

        # Inverse normal transformation (similar to MATLAB's norminv)
        sigma_values = norm.ppf(cdf_values)

        # Invert the y-axis for odd-indexed files if needed
        '''if index % 2 == 0:
            sigma_values = -sigma_values'''
        
        # Extract timestamp for labeling
        timestamp = table_name.split('/')[-1].split(' ')[0]

        # Assign color based on the group of four
        color = colors[(index // 4) % 20]
        
        '''if index % 4 == 2:
            if index // 4 != 0:
                # Plotting individual points as in MATLAB's script
                plt.scatter(sorted_data, sigma_values, label=timestamp, color=color, s=5)'''

        plt.scatter(sorted_data, sigma_values, label=timestamp, color=color, s=5)

    plt.xlabel('Conductance (S)')
    plt.ylabel('Sigma (Standard deviations)')
    plt.title('Transformed CDF of States')
    # Place the legend outside the graph to the left
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Place the legend outside the graph to the left and adjust subplot parameters
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(rect=[0, 0, 0.95, 1])  # Adjust the right boundary of the plot area  #adjust the third parameter

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()

    close_connection()  # Ensure to close the database connection when done

    return plot_data