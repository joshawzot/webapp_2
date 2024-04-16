from tools_for_plots import *


import io
import base64

def plot_tcr(group_data, selected_groups):
    # Map table names to temperatures for x-axis labeling
    table_names = ['01_25c_npy', '02_55c_npy', '03_85c_npy', '04_55c_npy']
    temperature_labels = ['25°C', '55°C', '85°C', 'Second 55°C']  # Assuming these are the temperatures represented by the table_names in order

    # Temperature differences for TCR calculations, assuming 25°C as the reference
    delta_ts = [0, 30, 60, 30]  # Corresponding ΔT for each table_name, assuming the last is another measurement at 55°C

    fig, ax = plt.subplots()

    # Assuming '01_25c_npy' contains G1 values for the reference temperature (25°C)
    idx_25c = table_names.index('01_25c_npy')

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
        ax.plot(temperature_labels[1:], tcr_values, marker='o', label=f'Group {group_idx}')

    ax.set_xlabel('Temperature Conditions')
    ax.set_ylabel('TCR')
    ax.legend(title="Selected Groups", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True)
    ax.set_title('Temperature Coefficient of Resistance (TCR) across Conditions')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()  # Close the plot to free memory
    buf.seek(0)  # Seek to the start of the bytes buffer
    encoded_image = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    return encoded_image

def generate_plot(table_names, database_name, form_data):
    # Ensure this connects to your database
    connection = create_connection(database_name)

    selected_groups = form_data.get('selected_groups', [])

    sub_array_size = int(form_data.get('sub_array_size', 16))  # Convert to int to ensure it's not a float

    # Initialize an empty list to hold the encoded plots
    encoded_plots = []
    group_data = []

    # Define a colormap
    cmap = cm.get_cmap('viridis', len(table_names))
    norm = mcolors.Normalize(vmin=0, vmax=len(table_names) - 1)
    colors = [cmap(norm(i)) for i in range(len(table_names))]

    # Process each table and collect data and statistics (only once)
    for table_name in table_names:
        groups, stats, array_size = get_group_data(table_name, selected_groups, database_name, sub_array_size)
        group_data.append(groups)

    encoded_plot_tcr = plot_tcr(group_data, selected_groups)
    encoded_plots.append(encoded_plot_tcr)        
    
    return encoded_plots

