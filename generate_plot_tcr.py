from tools_for_plots import *
from db_operations import create_connection, fetch_data, close_connection
import io
import base64
import numpy as np

def plot_tcr(group_data, selected_groups):
    # Map table names to temperatures for x-axis labeling
    table_names = ['25C', '55C', '85C', '125C','85C_2','55C_2']
    temperature_labels = ['25C', '55C', '85C', '125C','85C_2','55C_2']  # Assuming these are the temperatures represented by the table_names in order

    # Temperature differences for TCR calculations, assuming 25°C as the reference
    delta_ts = [0, 30, 60, 100, 60, 30]  # Corresponding ΔT for each table_name, assuming the last is another measurement at 55°C

    fig, ax = plt.subplots()

    # Assuming '01_25c_npy' contains G1 values for the reference temperature (25°C)
    idx_25c = table_names.index('25C')

    group_labels = ['36.20', '62.20', '88.16', '114.18']
    for group_idx in selected_groups:
        tcr_values = []

        # Extract G1 for each group from '01_25c_npy'
        g1 = np.mean(group_data[idx_25c][group_idx])

        for i, table_name in enumerate(table_names):
            if i == idx_25c:  # Skip the reference table itself
                continue
            
            # Extract G2 for current table
            g2 = np.mean(group_data[i][group_idx])

            # Calculate TCR for current condition
            if delta_ts[i] == 0:  # Avoid division by zero for the reference temperature
                tcr_value = 0
            else:
                tcr_value = (g1 - g2) / g2 / delta_ts[i]
            tcr_values.append(tcr_value)

        # Plotting TCR values for the current group
        #ax.plot(temperature_labels[1:], tcr_values, marker='o', label=f'Group {group_idx}')
        ax.plot(temperature_labels[1:], tcr_values, marker='o', label=group_labels[group_idx])


    ax.set_xlabel('Temperature')
    ax.set_ylabel('TCR')
    #ax.legend(title="Selected Groups", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.legend(title="Conductance (uS)", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True)
    #ax.set_title('Temperature Coefficient of Resistance (TCR) across Conditions')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()  # Close the plot to free memory
    buf.seek(0)  # Seek to the start of the bytes buffer
    encoded_image = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    return encoded_image

def reorder_tables(table_names):
    # Define the desired order
    desired_order = [
        'conductance_25c_csv',
        'conductance_55c_csv',
        'conductance_85c_csv',
        'conductance_125c_csv',
        'conductance_85c_2_csv',
        'conductance_55c_2_csv'
    ]
    
    # Create a dictionary to map table names to their desired positions
    order_map = {name: i for i, name in enumerate(desired_order)}
    
    # Reorder according to the desired sequence
    reordered_tables = sorted(table_names, key=lambda name: order_map.get(name, float('inf')))
    
    return reordered_tables

def generate_plot(table_names, database_name, form_data):
    # Ensure this connects to your database
    connection = create_connection(database_name)

    print("table_names:", table_names)
    reordered_table_names = reorder_tables(table_names)
    print("reordered_table_names:", reordered_table_names)
    selected_groups = form_data.get('selected_groups', [])
    print("selected_groups:", selected_groups)
    #selected_groups = list(range(4096))
    #selected_groups = list(range(82944))
    #return 0
    
    # Fetch sub_array_size from form_data, which could be either a string or a list
    sub_array_size_raw = form_data.get('sub_array_size', '324,64')  # Default to '324,64' if not present
    print("sub_array_size_raw:", sub_array_size_raw)
    sub_array_size = tuple(sub_array_size_raw)
    print("sub_array_size:", sub_array_size)

    # Initialize an empty list to hold the encoded plots
    encoded_plots = []
    group_data = []

    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.colors import ListedColormap
    import numpy as np

    n_colors = len(table_names)

    # Generate a list of colors from the 'viridis' colormap
    base_colors = plt.cm.viridis(np.linspace(0, 1, n_colors))

    # Create a new ListedColormap from the base colors
    cmap = ListedColormap(base_colors)

    # Define a normalization from values in [0, n_colors-1]
    norm = mcolors.Normalize(vmin=0, vmax=n_colors - 1)

    # Generate color values for each index
    colors = [cmap(norm(i)) for i in range(n_colors)]


    # Process each table and collect data and statistics (only once)
    for table_name in reordered_table_names:
        groups, stats, _, num_of_groups, selected_groups = get_group_data_new(table_name, selected_groups, database_name, sub_array_size)
        group_data.append(groups)

    encoded_plot_tcr = plot_tcr(group_data, selected_groups)
    encoded_plots.append(encoded_plot_tcr)       
    
    return encoded_plots

