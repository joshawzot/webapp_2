import pandas as pd
from io import BytesIO
import base64
import mysql.connector
import matplotlib.pyplot as plt
import os
from db_operations import create_connection, fetch_data, close_connection

def generate_plot_horizontal_boxplotsigma_xn(table_name, database_name):
    # Connect to the MySQL database
    connection = create_connection(database_name)

    '''start_value = form_data.get('start_value')
    end_value = form_data.get('end_value')
    left_y_axis_label = form_data.get('left_y_axis_label')
    right_y_axis_label = form_data.get('right_y_axis_label')
    x_axis_label = form_data.get('x_axis_label')'''
 
    start_value = None
    end_value = None
    left_y_axis_label = None
    right_y_axis_label = None
    x_axis_label = None
    
    # Fetch the data directly into a pandas DataFrame
    query = f"SELECT * FROM `{table_name}`"
    df = pd.read_sql(query, connection)
    
    if start_value is None:
        start_value = df.iloc[:, 0].min()
        
    if end_value is None:
        end_value = df.iloc[:, 0].max()
    
    filtered_df = df[df.iloc[:, 0].between(start_value, end_value)]
    x_values = filtered_df.iloc[:, 0].tolist()
    y_values = filtered_df.iloc[:, 1:].values

    std_values = y_values.std(axis=1)
    mean_values = y_values.mean(axis=1)
    average_sigma = std_values.mean()

    #fig, ax = plt.subplots()
    fig, ax = plt.subplots(figsize=(15, 10))  # Adjust the figure size here
    ax.boxplot(y_values.T, labels=x_values, showfliers=True)
    #ax.boxplot(y_values.T, labels=x_values, showfliers=False)
    
    ax_right = ax.twinx()
    #ax_right.plot(range(1, len(x_values) + 1), std_values, marker='s', color='blue', label='Sigma')
    ax_right.plot(range(1, len(x_values) + 1), std_values, marker='s', color=(0.6, 0.6, 1.0), label='Sigma')  # Lighter color here
    ax_right.axhline(average_sigma, linestyle='dashed', color='green', label='Avg Sigma')
    ax.plot(range(1, len(x_values) + 1), mean_values, marker='o', color='red', label='Mean')  # Plotting mean values trend line here

    #if left_y_axis_label:
        #ax.set_ylabel(left_y_axis_label, color='black', fontsize=15)
    ax.tick_params(axis='y', labelcolor='black')
        
    #if right_y_axis_label:
        #ax_right.set_ylabel(right_y_axis_label, color='blue', fontsize=15)
    ax_right.tick_params(axis='y', labelcolor='blue')
    
    #if x_axis_label:
        #ax.set_xlabel(x_axis_label, fontsize=15)
    ax_right.tick_params(axis='x', labelcolor='black')
    
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    close_connection()
    
    return plot_data