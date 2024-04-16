import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import mysql.connector
import os
import numpy as np
from db_operations import create_connection, fetch_data, close_connection
from matplotlib import colors

def generate_plot_checkerboard(table_name, database_name):
    # Connect to the MySQL database
    connection = create_connection(database_name)
    
    # Fetch the data directly into a pandas DataFrame
    query = f"SELECT * FROM `{table_name}`"
    df = pd.read_sql(query, connection)

    # Reshape the data into a 128x256 matrix
    matrix = df.values.reshape((128, 256))

    # Calculate the mean value for each 16x16 block
    block_means = []
    for row in range(8):
        for col in range(16):
            block = matrix[row*16:(row+1)*16, col*16:(col+1)*16]
            mean = block.mean()
            block_means.append(mean)

    # Reshape the block means into an 8x16 matrix
    board = np.array(block_means).reshape((8, 16))

    # Define a custom color map by modifying the 'Blues' colormap
    cmap = plt.get_cmap('Blues')
    new_colors = cmap(np.linspace(0, 0.8, cmap.N))  # Adjust the range to make the darkest blue lighter
    new_cmap = colors.ListedColormap(new_colors)

    # Plotting the board with mean values annotated on each block
    fig, ax = plt.subplots(figsize=(15, 10))  # Adjust the figure size here
    im = ax.imshow(board, cmap=new_cmap)

    # Add mean values as text annotations
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            text = ax.text(j, i, f'{board[i, j]:.2f}', ha='center', va='center', color='black', fontsize=10)

    # Set up colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Mean Values', rotation=-90, va="bottom")

    # Mark the horizontal axis with ticks for 1-based block numbers
    ax.set_xticks(np.arange(board.shape[1]))
    ax.set_xticklabels(np.arange(1, board.shape[1]+1))

    # Mark the vertical axis with ticks for 1-based block numbers
    ax.set_yticks(np.arange(board.shape[0]))
    ax.set_yticklabels(np.arange(1, board.shape[0]+1))

    # Add horizontal and vertical border lines for each block
    for i in range(1, board.shape[0]):
        ax.axhline(i - 0.5, color='black')
    for j in range(1, board.shape[1]):
        ax.axvline(j - 0.5, color='black')
        
    plt.title('Mean Values of each block(16x16 cells)')

    # Save the plot to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    
    # Convert the BytesIO object to a base64 encoded string
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    close_connection()
    
    return plot_data