import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import mysql.connector
import os
from db_operations import create_connection, fetch_data, close_connection

def generate_plot_vertical_xn(table_name, database_name):
    # Connect to the MySQL database
    connection = create_connection(database_name)
    
    # Fetch the data directly into a pandas DataFrame
    query = f"SELECT * FROM `{table_name}`"
    df = pd.read_sql(query, connection)

    # Extract the x-axis data
    x_data = df.iloc[:, 0]

    # Extract the y-axis data columns (excluding the first column)
    y_data = df.iloc[:, 1:]

    # Plotting
    plt.figure(figsize=(10, 6))

    for column in y_data.columns:
        plt.plot(x_data, y_data[column], label=column)

    #plt.xlabel(x_axis_label, fontsize=15)  # x-axis label
    #plt.ylabel(left_y_axis_label, fontsize=15)   # y-axis label
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    plt.legend(fontsize=15)
    plt.grid(True)

    # Save the plot to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    
    # Convert the BytesIO object to a base64 encoded string
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    close_connection()
    
    return plot_data