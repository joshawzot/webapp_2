import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from io import BytesIO
import base64
import matplotlib
from db_operations import create_connection, fetch_data, close_connection

matplotlib.use('Agg')

def process_file(table, df, forming_voltages_by_dut):   
    match = re.search(r'die(\d+)', table, re.I)
    if not match:
        print('not match')
        return

    coordinate = int(match.group(1))
    dut = int(re.search(r'DUT(\d+)', table, re.I).group(1))
    
    #print('coordinate:', coordinate)
    #print('dut:', dut)
    #print(df)  #hok here
    voltage = df.iloc[:, 0].astype(float).values
    current = df.iloc[:, 1].astype(float).values
    #print('voltage:', voltage)
    #print('current:', current)
    # Finding the forming voltage
    for i in range(1, len(current)):
        if (voltage[i] > 1) and (voltage[i] < 3) and (current[i] > 3 * current[i-1]):
            print("found forming voltage")
            if dut not in forming_voltages_by_dut:
                forming_voltages_by_dut[dut] = {}
            if coordinate not in forming_voltages_by_dut[dut]:
                forming_voltages_by_dut[dut][coordinate] = []
                
            forming_voltages_by_dut[dut][coordinate].append(float(voltage[i]))
            print('forming_voltage_by_dut:', forming_voltages_by_dut)
            break
            
def plot_colormap(forming_voltages, mean_std_info, dut):
    rows = [
        (65, 69), (64, 57), (47, 56), (46, 37), (27, 36),
        (26, 17), (9, 16), (8, 3), (1, 2)
    ]

    # Create a blank matrix with zeros
    matrix = np.full((9, 10), np.nan)

    for row, (start, end) in enumerate(rows):
        if (row + 1) % 2 == 0:  # even rows (right to left)
            coords = list(range(start, end - 1, -1))
        else:  # odd rows (left to right)
            coords = list(range(start, end + 1))
        
        centering_offset = (10 - len(coords)) // 2

        for col_offset, coord in enumerate(coords):
            col_idx = centering_offset + col_offset
            matrix[row, col_idx] = forming_voltages.get(coord, np.nan)

    # Plotting the colormap
    fig, ax = plt.subplots()
    #cax = ax.imshow(matrix, cmap='viridis', interpolation='nearest', aspect='auto', vmin=0)
    cax = ax.imshow(matrix, cmap='viridis', interpolation='nearest', aspect='auto', vmin=2, vmax=4) # Set vmin=2 and vmax=4
    fig.colorbar(cax, label='Forming Voltage')
    ax.set_title('Forming Voltage Colormap for DUT' + str(dut))

    # Annotate the colormap with coordinates
    for row, (start, end) in enumerate(rows):
        if (row + 1) % 2 == 0:  # even rows (right to left)
            coords = list(range(start, end - 1, -1))
        else:  # odd rows (left to right)
            coords = list(range(start, end + 1))
        
        centering_offset = (10 - len(coords)) // 2

        for col_offset, coord in enumerate(coords):
            col_idx = centering_offset + col_offset
            ax.text(col_idx, row, str(coord), ha='center', va='center', color='black' if matrix[row, col_idx] > 0 else 'black')
            
            # Check if the value at the coordinate is not nan before annotating with forming voltage value
            if not np.isnan(matrix[row, col_idx]):
                ax.text(col_idx, row + 0.25, "{:.2f}".format(round(matrix[row, col_idx], 2)), ha='center', va='center', fontsize=5, color='black')
    # Draw a circle around the colormap
    center = (matrix.shape[1] / 2 - 0.5, matrix.shape[0] / 2 - 0.5)
    radius = max(matrix.shape[0], matrix.shape[1]) / 2 + 0.5
    circle = patches.Circle(center, radius=radius, edgecolor='black', facecolor='none')
    ax.add_patch(circle)

    # Display mean and std on the colormap
    ax.text(0, 8.5, f"Mean: {mean_std_info['mean']:.2f}", ha='center', va='center', color='black', fontsize=10)
    ax.text(0, 9.5, f"Sigma: {mean_std_info['std']:.2f}", ha='center', va='center', color='black', fontsize=10)   
    ax.axis('off')

    # Save the plot to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    # Convert the BytesIO object to a base64 encoded string
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()

    return plot_data

def generate_plot_forming_voltage_map(table_names, database_name):
    # Connect to the MySQL database
    connection = create_connection(database_name)
    
    forming_voltages_by_dut = {}
    for table in table_names:
        query = f"SELECT * FROM `{table}`"
        df = pd.read_sql(query, connection)
        print(df) #ok here

        process_file(table, df, forming_voltages_by_dut)
        
    for dut, coordinates in forming_voltages_by_dut.items():
        for coordinate, values in coordinates.items():
            if values:
                forming_voltages_by_dut[dut][coordinate] = np.nanmean(values)
            else:
                forming_voltages_by_dut[dut][coordinate] = np.nan

    
    means_std_by_dut = {}
    for dut, coordinates in forming_voltages_by_dut.items():
        values = list(coordinates.values())
        mean_val = np.nanmean(values)
        std_val = np.nanstd(values)
        means_std_by_dut[dut] = {"mean": mean_val, "std": std_val}

    plots_by_dut = []
    for dut, forming_voltages in forming_voltages_by_dut.items():
        plot_data = None
        plot_data = plot_colormap(forming_voltages, means_std_by_dut[dut], dut)
        plots_by_dut.append(plot_data)  # Update the dictionary with plot_data for each DUT
        print('dut:', dut)
    
    #print('length:', len(plots_by_dut))   
    close_connection()
    return plots_by_dut