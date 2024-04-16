import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import mysql.connector
import os
from db_operations import create_connection, fetch_data, close_connection
from matplotlib.ticker import MaxNLocator, ScalarFormatter

#def generate_plot_horizontal_boxplotsigma_xnxm(table_names, database_name, form_data):
def generate_plot_horizontal_boxplotsigma_xnxm(table_names, database_name):
    # Connect to the MySQL database
    connection = create_connection(database_name)
    print("table_names")
    print(table_names)

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
    
    # List to store dataframes
    dfs = []
    
    for table in table_names:
        print("table")
        print(table)
        query = f"SELECT * FROM `{table}`"
        df = pd.read_sql(query, connection)
        #print(df)

        # Append the dataframe to the list
        dfs.append(df)

    # Take the first column from the first table
    first_column = dfs[0].iloc[:, :1]

    # Concatenate first column with remaining columns from all tables
    remaining_columns = [df.iloc[:, 1:] for df in dfs]
    df = pd.concat([first_column] + remaining_columns, axis=1)
    
    print(df)
    
    # Delete columns if any cell in that column has value smaller than 1e-5
    #cols_to_drop = [col for col in df.columns if (df[col] < 1e-5).any()]
    cols_to_drop = df.columns[df.apply(lambda col: (col < 1e-5).any(), axis=0)]
    df.drop(columns=cols_to_drop, inplace=True)

    # Delete the last cell for each column (which means deleting the last row)
    #df.drop(data_df.tail(1).index, inplace=True)  # drop last row

    if start_value is None:
        start_value = df.iloc[:, 0].min()
        
    if end_value is None:
        end_value = df.iloc[:, 0].max()
    
    filtered_df = df[df.iloc[:, 0].between(start_value, end_value)]
    x_values = filtered_df.iloc[:, 0].tolist()
    y_values = filtered_df.iloc[:, 1:].values
    
    # Extract x_values from the first column of the DataFrame
    #filtered_df = df[df.iloc[:, 0].between(start_value, end_value)]
    #filtered_df = df[df.iloc[:, 0].between(0.4, 1.3)]
    #x_values = filtered_df.iloc[:, 0].tolist()
    #y_values = filtered_df.iloc[:, 1:].values

    # Calculate the standard deviation and mean for each set of y_values
    std_values = y_values.std(axis=1)
    mean_values = y_values.mean(axis=1)

    # Calculate the average sigma
    average_sigma = std_values.mean()

    # Create a box plot using matplotlib
    fig, ax = plt.subplots(figsize=(15, 10))  # Adjust the figure size here
    ax.boxplot(y_values.T, labels=x_values, showfliers=True)

    # Create a twin axes instance on the right side
    ax_right = ax.twinx()

    # Plot the std line on the right y-axis
    ax_right.plot(range(1, len(x_values) + 1), std_values, marker='s', color='blue', label='Sigma')
    ax_right.axhline(average_sigma, linestyle='dashed', color='green', label='Avg Sigma')

    # Set the label and color for the right y-axis
    ax_right.set_ylabel(right_y_axis_label, color='blue', fontsize=15)
    ax_right.tick_params(axis='y', labelcolor='blue')

    # Set the label and color for the left y-axis
    ax.set_ylabel(left_y_axis_label, color='black', fontsize=15)
    ax.tick_params(axis='y', labelcolor='black')

    # Add mean lines
    ax.plot(range(1, len(x_values) + 1), mean_values, marker='o', color='red', label='Mean', markersize=3)

    ax.set_xlabel(x_axis_label, fontsize=15)
    ax.set_xticks(range(1, len(x_values) + 1, 2))  # Only show every second tick
    rounded_x_values = ["{:.2f}".format(value) for value in x_values[::2]]
    ax.set_xticklabels(rounded_x_values, rotation=45, fontsize=10)  # Show the corresponding labels for the selected ticks
    plt.yticks(fontsize=10)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=10, prune='both'))  # Choose a reasonable value for nbins

    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax_right.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    yticks_left = ax.get_yticks()
    formatted_yticks_left = ['{:.0f}'.format(ytick * 1e6) for ytick in yticks_left]
    ax.set_yticklabels(formatted_yticks_left, fontsize=10)

    ax_right.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    yticks_right = ax_right.get_yticks()
    formatted_yticks_right = ['{:.0f}'.format(ytick * 1e6) for ytick in yticks_right]
    ax_right.set_yticklabels(formatted_yticks_right)

    lines, labels = ax.get_legend_handles_labels()
    lines_right, labels_right = ax_right.get_legend_handles_labels()

    formatted_average_sigma = '{:.2f}'.format(average_sigma * 1e6)
    labels_right[-1] += f': {formatted_average_sigma}'

    ax.legend(lines + lines_right, labels + labels_right, loc='upper left', fontsize=10)

    # Save the plot to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    # Convert the BytesIO object to a base64 encoded string
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    close_connection()
    
    return plot_data