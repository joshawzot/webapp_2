import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import mysql.connector
import os
from db_operations import create_connection, fetch_data, close_connection

def generate_plot_horizontal_sigma_xnxm(table_names, database_name):
    # Connect to the MySQL database
    connection = create_connection(database_name)
    
    plt.figure(figsize=(10, 6))

    for table in table_names:
        query = f"SELECT * FROM `{table}`"
        df = pd.read_sql(query, connection)

        x_data = df.iloc[:, 0]
        y_data = df.iloc[:, 1:]
        stds = y_data.std(axis=1) * 1e6
        mean_std = stds.mean()  # Calculate the mean of the standard deviations
        
        label_text = f"{table} (Mean: {mean_std:.2f} uS)"  # Create label with mean value
        plt.plot(x_data, stds, label=label_text, marker='s', linestyle='-')

    #plt.xlabel(x_axis_label, fontsize=16)  # Adjust this as needed
    #plt.ylabel(left_y_axis_label, fontsize=16)  # Adjust this as needed
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)

    # Save the plot to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    # Convert the BytesIO object to a base64 encoded string
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    close_connection()
    
    return plot_data