import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from io import BytesIO
import base64
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from db_operations import create_connection, fetch_data, close_connection
from sklearn.linear_model import LinearRegression
from scipy.stats import norm

# Set print options
np.set_printoptions(threshold=np.inf)

# Constants
FILTER_ZEROS = False
FONT_SIZE = 12

REFL = 127
REFL_ADC_READ = 49
ADC_levels = 255
DAC_levels = 255
Vdd = 3.3
TIA_GAIN_VALUE = 1e3, 2.5e3, 1e4, 4e4
TIA_GAIN_READ = 3
SRREF2 = 139

def dac2v(dac_code, DAC_levels, Vdd):
    return dac_code / DAC_levels * Vdd

def v2i(v, TIA_GAIN_VALUE, TIA_GAIN_READ):
    return v / TIA_GAIN_VALUE[TIA_GAIN_READ]

def i2g(i_read, SRREF2, REFL, DAC_levels, Vdd):
    v_read = dac2v(SRREF2 - REFL, DAC_levels, Vdd)
    return i_read / v_read

def adc2v(adc_code, REFL, REFL_ADC_READ, ADC_levels, DAC_levels, Vdd):
    Vrefl = dac2v(REFL, DAC_levels, Vdd)
    Vadc_low_ref = dac2v(REFL_ADC_READ, DAC_levels, Vdd)
    return (1 - adc_code / ADC_levels) * (Vrefl - Vadc_low_ref)

def adc2g(adc_code, REFL, REFL_ADC_READ, ADC_levels, DAC_levels, Vdd, TIA_GAIN_VALUE, TIA_GAIN_READ, SRREF2):
    v_device = adc2v(adc_code, REFL, REFL_ADC_READ, ADC_levels, DAC_levels, Vdd)
    i_read = v2i(v_device, TIA_GAIN_VALUE, TIA_GAIN_READ)
    return i2g(i_read, SRREF2, REFL, DAC_levels, Vdd)

def compute_chunk_statistics(chunk):
    if FILTER_ZEROS:
        masked_chunk = np.ma.masked_equal(chunk, 0)
        average = np.ma.mean(masked_chunk)
        std_dev = np.ma.std(masked_chunk)
    else:
        average = np.mean(chunk)
        std_dev = np.std(chunk)
    return average, std_dev

def get_statistics(section_data_reshaped, side_of_a_square, side_subsquare):
    averages = np.zeros((int(side_of_a_square/side_subsquare), int(side_of_a_square/side_subsquare)))
    std_devs = np.zeros((int(side_of_a_square/side_subsquare), int(side_of_a_square/side_subsquare)))

    for i in range(int(side_of_a_square/side_subsquare)):
        for j in range(int(side_of_a_square/side_subsquare)):
            chunk = section_data_reshaped[i*side_subsquare:(i+1)*side_subsquare, j*side_subsquare:(j+1)*side_subsquare]
            avg, std = compute_chunk_statistics(chunk)
            averages[i, j] = avg
            std_devs[i, j] = std
    
    return averages, std_devs

def plot_averages_table(averages, std_devs, ax, cbar_ax, vmin, vmax):
    cax = ax.matshow(averages, cmap=plt.cm.YlGnBu, vmin=vmin, vmax=vmax, aspect='equal')
    
    for i in range(averages.shape[0]):
        for j in range(averages.shape[1]):
            text = f'{averages[i, j]:.2f}\n({std_devs[i, j]:.2f})'
            ax.text(j, i, text, ha='center', va='center', color='black', fontsize=FONT_SIZE)

    plt.colorbar(cax, cax=cbar_ax)

def save_figure(fig, plot_data_list):
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    plot_data_list.append(plot_data)
    buf.close()
    plt.close(fig)

def plot_section(df, start_index, num_cells, side_of_a_square, side_subsquare, min_val, max_val, gs, cmap, section_numbers, all_zero_coords):
    fig = plt.figure(figsize=(40, 40))
    section_data = df.iloc[start_index:start_index+num_cells, 0].values
    section_data_reshaped = section_data.reshape((side_of_a_square, side_of_a_square))

    # At the point in the code where you calculate zero coordinates: (entire area)
    #zero_coords = np.column_stack(np.where(section_data_reshaped == 0))
    #all_zero_coords.extend(zero_coords.tolist())  # Add the zero coordinates to the all_zero

    for section_number in section_numbers: #(only the selected section number)
        x_coord, y_coord = divmod(int(section_number), int(side_of_a_square // side_subsquare))
        chunk = section_data_reshaped[int(x_coord) * side_subsquare:(int(x_coord) + 1) * side_subsquare, int(y_coord) * side_subsquare:(int(y_coord) + 1) * side_subsquare]

        # Calculate and collect zero coordinates for the specific chunk (subsection)
        zero_coords_chunk = np.column_stack(np.where(chunk == 0))

        # Adjust these coordinates relative to the whole image
        zero_coords_chunk[:, 0] += x_coord * side_subsquare  # Adjust x coordinates
        zero_coords_chunk[:, 1] += y_coord * side_subsquare  # Adjust y coordinates

        all_zero_coords.extend(zero_coords_chunk.tolist())
        
    # Define a new GridSpec layout with 4 rows and 3 columns
    gs = gridspec.GridSpec(4, 3, width_ratios=[1, 1, 0.05])

    # Plot the main section data and averages
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(section_data_reshaped, cmap=cmap, aspect='equal', vmin=min_val, vmax=max_val)
    ax0.tick_params(axis='both', which='major', labelsize=FONT_SIZE)

    ax1 = fig.add_subplot(gs[0, 1])
    cbar_ax = fig.add_subplot(gs[0, 2])
    averages, std_devs = get_statistics(section_data_reshaped, side_of_a_square, side_subsquare)
    plot_averages_table(averages, std_devs, ax1, cbar_ax, vmin=min_val, vmax=max_val)
    ax1.tick_params(axis='both', which='major', labelsize=FONT_SIZE)

    for label in cbar_ax.xaxis.get_ticklabels() + cbar_ax.yaxis.get_ticklabels():
        label.set_fontsize(FONT_SIZE)

    # Add code for CDF plot
    ax_cdf = fig.add_subplot(gs[1, 0:2])
    print(f"Number of sections: {len(section_data_reshaped)}")
    print(f"Section numbers: {section_numbers}")
    for section_number in section_numbers:
        print(f"Processing section number: {section_number}")
        x_coord, y_coord = divmod(int(section_number), int(side_of_a_square // side_subsquare))
        print("A")
        chunk = section_data_reshaped[int(x_coord)*side_subsquare:int(x_coord+1)*side_subsquare, int(y_coord)*side_subsquare:int(y_coord+1)*side_subsquare]
        print("B")
        filtered_chunk = chunk.flatten() if not FILTER_ZEROS else chunk[chunk != 0].flatten()
        print("C")
        counts, bin_edges = np.histogram(filtered_chunk, bins=256, density=True)
        print("D")
        cdf = np.cumsum(counts)
        print("E")
        ax_cdf.plot(bin_edges[1:], cdf / cdf[-1], label=f'Section {section_number}')  # Normalized CDF
        print("F")
    
    ax_cdf.set_xlim(0, 255)
    ax_cdf.legend(fontsize=FONT_SIZE)
    ax_cdf.set_xlabel('Value')
    ax_cdf.set_ylabel('Cumulative Distribution')

    # Add code for Histogram with BER
    ax_hist = fig.add_subplot(gs[2, 0:2])
    midpoint = (min_val + max_val) / 2
    for section_number in section_numbers:
        x_coord, y_coord = divmod(int(section_number), int(side_of_a_square // side_subsquare))
        print("G")
        chunk = section_data_reshaped[int(x_coord)*side_subsquare:int(x_coord+1)*side_subsquare, int(y_coord)*side_subsquare:int(y_coord+1)*side_subsquare]
        print("H")
        filtered_chunk = chunk.flatten() if not FILTER_ZEROS else chunk[chunk != 0].flatten()
        print("I")

        # Calculate the amount and percentage of 0 values
        zero_count = np.sum(chunk == 0)
        total_count = chunk.size
        zero_percentage = zero_count / total_count * 100  # Percentage of zeros

        # Explicitly define the bin edges from 0 to 256, ensuring each bin represents an integer value
        bin_edges = np.arange(-0.5, 256 + 0.5, 1)  # 256 bins, from -0.5 to 255.5
        print("J")

        # Calculate histogram using the defined bin edges
        counts, _ = np.histogram(filtered_chunk, bins=bin_edges)
        print("K")

        # Calculate BER
        lower_half_count = np.sum(counts[bin_edges[:-1] + 0.5 < midpoint])
        upper_half_count = np.sum(counts[bin_edges[:-1] + 0.5 >= midpoint])
        ber = min(lower_half_count, upper_half_count) / np.sum(counts) * 100  # BER as a percentage
        #label = f'Section {section_number} (BER: {ber:.2f}%, Zeros: {zero_count}, {zero_percentage:.2f}%)'
        label = f'Section {section_number}, (Zeros count: {zero_count}, {zero_percentage:.2f}%)'
        
        ax_hist.bar(bin_edges[:-1] + 0.5, counts, width=1, alpha=0.5, label=label)
        print("L")

    ax_hist.set_xlim(-10, 255)
    ax_hist.set_yscale('log')
    ax_hist.legend(fontsize=FONT_SIZE)
    ax_hist.set_xlabel('Value')
    ax_hist.set_ylabel('Frequency')

    # Add code for Boxplots
    ax_box = fig.add_subplot(gs[3, 0:2])
    boxplot_data = []
    print("M")
    for section_number in section_numbers:
        x_coord, y_coord = divmod(int(section_number), int(side_of_a_square // side_subsquare))
        chunk = section_data_reshaped[int(x_coord)*side_subsquare:int(x_coord+1)*side_subsquare, int(y_coord)*side_subsquare:int(y_coord+1)*side_subsquare]
        filtered_chunk = chunk.flatten() if not FILTER_ZEROS else chunk[chunk != 0].flatten()
        boxplot_data.append(filtered_chunk)
    print("N")

    ax_box.set_xlim(0, 255)
    ax_box.boxplot(boxplot_data, vert=False, showmeans=True)
    ax_box.set_yticklabels([f'Section {sn}' for sn in section_numbers])
    ax_box.set_xlabel('Value')
    print("O")
    fig.tight_layout()
    return fig

def generate_plot_endurance(table_names, database_name, form_data):
    combined_plot_data_list = []  # List to store data for all plots
    FONT_SIZE = 12  # Define a default font size for plots

    # Loop through each table name provided
    for table_name in table_names:
        print('Processing table:', table_name)
        connection = create_connection(database_name)

        # Fetch data from the current table
        df = pd.read_sql(f"SELECT * FROM `{table_name}`", connection) 
        print(df)

        df = adc2g(df, REFL, REFL_ADC_READ, ADC_levels, DAC_levels, Vdd, TIA_GAIN_VALUE, TIA_GAIN_READ, SRREF2) * 1e6
        print(df)

        # Use the helper function to process form data
        '''side_of_a_square = int(form_data.get('side_of_a_square', 0))  # Default to 0 if not found
        side_subsquare = int(form_data.get('side_subsquare', 0))
        section_numbers = form_data.get('section_numbers', [])'''

        # Use the helper function to process form data
        side_of_a_square = 256
        side_subsquare = 32
        section_numbers = [0,1,2,3,8,9,10,11,16,17,18,19,24,25,26,27]

        combined_data = {'boxplot': [], 'cdf': [], 'histogram': []}

        num_cells = side_of_a_square ** 2
        num_sets = int(len(df) / num_cells)

        cmap = plt.cm.get_cmap('YlGnBu')
        min_val = 0
        max_val = 255

        plot_data_list = []
        all_zero_coords = []

        for i in range(num_sets):
            gs = gridspec.GridSpec(4, 3, width_ratios=[1, 1, 0.05])
            fig = plot_section(df, i * num_cells, num_cells, side_of_a_square, side_subsquare, min_val, max_val, gs, cmap, section_numbers, all_zero_coords)
            print("P")
            save_figure(fig, plot_data_list)

        combined_plot_data_list.extend(plot_data_list)  # Collect all plot data
        close_connection()  # Ensure the connection is correctly closed with its instance

    return combined_plot_data_list

