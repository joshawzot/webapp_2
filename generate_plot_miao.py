'''import numpy as np
import matplotlib.pyplot as plt
import os

# Function to calculate cosine similarity
def calculate_cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# Clear all
plt.close('all')

# Paths
data_folder = os.getcwd()
vwl_filename = os.path.join(data_folder, 'vwl_voltage_applied_history.npy')
finetune_filename = os.path.join(data_folder, 'Read After Finetune0_conductance.npy')
history_filename = os.path.join(data_folder, 'read_history.npy')
read_after_finetune = np.load(finetune_filename)
print("read_after_finetune:", read_after_finetune)
print(read_after_finetune.shape)
#exit()

# Load and process data
vwl = np.load(vwl_filename)
print(vwl.shape)
print(vwl)
#exit()
read_history = np.squeeze(np.load(history_filename))

# Get dimensions and reshape
NUM_BL, NUM_WL, NUM_ITER = read_history.shape
vwl = np.squeeze(vwl)
print(vwl.shape)
print(vwl)
#exit()
enable_process_nan = 1
target = np.array([40, 80, 128, 160])
target_error_tol = 3
wl_per_target = 324
read_outliner_threshold = target_error_tol*3 
target_wl_st = np.array([1, wl_per_target+1, wl_per_target*2+1, wl_per_target*3+1])
target_wl_end = np.array([wl_per_target, wl_per_target*2, wl_per_target*3, wl_per_target*4])
#print("target_wl_st", target_wl_st)
#print("target_wl_end", target_wl_end)

#exit()
Vwl_min = np.nanmin(vwl)
Vwl_max = np.nanmax(vwl)
print('Vwl_min:', Vwl_min)
print('Vwl_max:', Vwl_max)
#print(vwl)
#exit()
# Initialize processed arrays
processed_read_hist = np.zeros((NUM_BL, NUM_WL, NUM_ITER))
processed_vwl_hist = np.zeros((NUM_BL, NUM_WL, NUM_ITER))  # For cosine similarity calculation only

print(vwl)

# Assuming NUM_BL and NUM_WL are defined somewhere in your code
for i in range(NUM_BL):
    for j in range(NUM_WL):
        cell_hist = np.squeeze(read_history[i, j, :])
        nan_index = np.where(np.isnan(cell_hist))[0]
        if nan_index.size > 0:
            first_nan_index = nan_index[0]
            cell_hist[first_nan_index:] = cell_hist[first_nan_index - 1]
        processed_read_hist[i, j, :] = cell_hist
        
        # Keep NaNs intact in vwl_temp
        vwl_temp = np.squeeze(vwl[i, j, :])
        processed_vwl_hist[i, j, :] = vwl_temp  # Direct assignment without NaN modification
        # Optionally print vwl_temp for debugging
        # print(vwl_temp)

# Optional debugging print
print(vwl)
#exit()

if enable_process_nan:
    read_history = processed_read_hist

# Assuming 'target' and other necessary variables are defined somewhere in your code
mean_G_hist = np.zeros((len(target), NUM_ITER))
mean_vwl_hist = np.zeros((len(target), NUM_ITER))
std_G_hist = np.zeros((len(target), NUM_ITER))
std_vwl_hist = np.zeros((len(target), NUM_ITER))

#print(vwl)
#exit()
for i in range(4):  # Adjust this range if necessary to match MATLAB's 1:4
    G_temp = read_history[0:NUM_BL, target_wl_st[i]-1:target_wl_end[i], 0:NUM_ITER]
    G_temp = np.reshape(G_temp, (NUM_BL * wl_per_target, NUM_ITER))
    #print(vwl)
    vwl_temp = vwl[0:NUM_BL, target_wl_st[i]-1:target_wl_end[i], 0:NUM_ITER]
   #print(vwl_temp)
    #exit()
    vwl_temp = np.reshape(vwl_temp, (NUM_BL * wl_per_target, NUM_ITER))

    print(vwl_temp.shape)  # Displaying the dimensions of vwl_temp
    #print(vwl_temp)
    #print(vwl_temp[0, :]) #first row
    #print(vwl_temp[:, 0])  # First column
    print(vwl_temp[:100, 0])  # First 100 rows of the first column

   
    #exit()
    mean_G = np.nanmean(G_temp, axis=0)
    std_G = np.nanstd(G_temp, axis=0)
    mean_vwl = np.nanmean(vwl_temp, axis=0)
    print(mean_vwl.shape)
    print(mean_vwl)

    #exit()
    std_vwl = np.nanstd(vwl_temp, axis=0)

    mean_G_hist[i, :] = mean_G
    std_G_hist[i, :] = std_G
    mean_vwl_hist[i, :] = mean_vwl
    std_vwl_hist[i, :] = std_vwl

#exit()
enable_figure = [1] * 14

# Plot mean conductance vs. Program iteration for each target
if enable_figure[1]:
    plt.figure(1)
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.plot(mean_G_hist[i, :], linewidth=2)
        plt.axhline(y=target[i], color='r', linestyle='--')  # Horizontal line at target value
        plt.title(f'Target = {target[i]} uS')
        plt.xlabel('Programm iter.')
        plt.ylabel('Mean conductance (uS)')
        plt.grid(True)
    plt.tight_layout()
    plt.savefig('Plot_Mean_Conductance.png')

# Plot std conductance vs. Program iteration for each target
if enable_figure[2]:
    plt.figure(2)
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.plot(std_G_hist[i, :], linewidth=2)
        plt.title(f'Target = {target[i]} uS')
        plt.xlabel('Programm iter.')
        plt.ylabel('std. conductance (uS)')
        plt.grid(True)
    plt.tight_layout()
    plt.savefig('Plot std conductance vs. Program iteration for each target.png')

print("mean_vwl_hist[i, :]:", mean_vwl_hist[i, :])
# Plot mean Vwl vs. Program iteration for each target
if enable_figure[3]:
    plt.figure(3)
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.plot(mean_vwl_hist[i, :], linewidth=2)
        plt.title(f'Target = {target[i]} uS')
        plt.xlabel('Programm iter.')
        plt.ylabel('Mean Vwl (V)')
        plt.grid(True)
    plt.tight_layout()
    plt.savefig(' Plot mean Vwl vs. Program iteration for each target.png')

# Plot std Vwl vs. Program iteration for each target
if enable_figure[4]:
    plt.figure(4)
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.plot(std_vwl_hist[i, :], linewidth=2)
        plt.title(f'Target = {target[i]} uS')
        plt.xlabel('Programm iter.')
        plt.ylabel('std Vwl (V)')
        plt.grid(True)
    plt.tight_layout()
    plt.savefig(' Plot std Vwl vs. Program iteration for each target.png')
    
# %% Check finetune disturbance - overview:
if enable_figure[5]:  # imagesc view.
    plt.figure(5)
    climit = [-20, 20]
    diff = read_after_finetune - np.squeeze(processed_read_hist[:, :, -1])
    plt.imshow(diff, cmap='viridis', vmin=climit[0], vmax=climit[1])
    title_str = 'G diff = Read after finetune - last fine tune read'
    plt.title(title_str)
    plt.colorbar()
    plt.clim(climit)
    plt.savefig(' Check finetune disturbance - overview:.png')

# %% Histogram
if enable_figure[6]:
    plt.figure(6)
    diff = read_after_finetune - np.squeeze(processed_read_hist[:, :, -1])
    plt.hist(diff.ravel(), bins=100)  # Flatten the array and specify 100 bins
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)
    title_str = f'Mean = {mean_diff:.2f}; std = {std_diff:.2f}'
    plt.title(title_str)
    plt.savefig(' Histogram.png')
    
# Check if the figure should be enabled
if enable_figure[7]:
    plt.figure(7)
    diff = read_after_finetune - np.squeeze(processed_read_hist[:, :, -1])
    read_outliner = diff > read_outliner_threshold
    read_outliner_per_WL = np.sum(read_outliner, axis=0)
    read_outliner_per_BL = np.sum(read_outliner, axis=1)

    # Plotting the outliers per WL
    plt.subplot(1, 2, 1)
    plt.scatter(np.arange(1, 1297), read_outliner_per_WL, marker='o')  # Assuming 1296 WLs
    plt.xlabel('WL index')
    plt.ylabel(f'Read outliner number (> {read_outliner_threshold} uS)')
    plt.title('Outliers per WL')
    
    # Plotting the outliers per BL
    plt.subplot(1, 2, 2)
    plt.scatter(np.arange(1, 65), read_outliner_per_BL, marker='o')  # Assuming 64 BLs
    plt.xlabel('BL index')
    plt.ylabel(f'Read outliner number (> {read_outliner_threshold} uS)')
    plt.title('Outliers per BL')

    plt.tight_layout()
    plt.savefig('last.png')
'''
import io
import base64
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from db_operations import *

def generate_plot_miao(table_names, database_name):
    # Create a database connection
    engine = create_engine(database_name)
    encoded_plots = []

    # Load data from the specified tables in the database
    # Filter and load data for 'read_after_finetune' that contains 'miao'
    read_after_finetune_table = next((name for name in table_names if "read_after_finetune" in name and "miao" in name), None)
    if read_after_finetune_table:
        read_after_finetune = pd.read_sql(read_after_finetune_table, con=engine).to_numpy()
    else:
        raise ValueError("No suitable table found for 'read_after_finetune' containing 'miao'")

    # Filter and load data for 'vwl' that contains 'miao'
    vwl_table = next((name for name in table_names if "vwl" in name and "miao" in name), None)
    if vwl_table:
        vwl = pd.read_sql(vwl_table, con=engine).to_numpy()
    else:
        raise ValueError("No suitable table found for 'vwl' containing 'miao'")

    # Filter and load data for 'read_history' that contains 'miao'
    read_history_table = next((name for name in table_names if "read_history" in name and "miao" in name), None)
    if read_history_table:
        read_history = pd.read_sql(read_history_table, con=engine).to_numpy()
    else:
        raise ValueError("No suitable table found for 'read_history' containing 'miao'")

    # Process data
    NUM_BL, NUM_WL, NUM_ITER = read_history.shape
    enable_process_nan = 1
    target = np.array([40, 80, 128, 160])
    target_error_tol = 3
    wl_per_target = 324
    read_outliner_threshold = target_error_tol * 3

    # Initialize processed arrays
    processed_read_hist = np.zeros((NUM_BL, NUM_WL, NUM_ITER))
    processed_vwl_hist = np.zeros((NUM_BL, NUM_WL, NUM_ITER))  # For cosine similarity calculation only

    for i in range(NUM_BL):
        for j in range(NUM_WL):
            cell_hist = np.squeeze(read_history[i, j, :])
            nan_index = np.where(np.isnan(cell_hist))[0]
            if nan_index.size > 0:
                first_nan_index = nan_index[0]
                cell_hist[first_nan_index:] = cell_hist[first_nan_index - 1]
            processed_read_hist[i, j, :] = cell_hist
            vwl_temp = np.squeeze(vwl[i, j, :])
            processed_vwl_hist[i, j, :] = vwl_temp

    if enable_process_nan:
        read_history = processed_read_hist

    # Generate all plots as per the existing code structure and save them as base64 strings
    figures = [
        'Plot_Mean_Conductance.png', 'Plot std conductance vs. Program iteration for each target.png',
        'Plot mean Vwl vs. Program iteration for each target.png', 'Plot std Vwl vs. Program iteration for each target.png',
        'Check finetune disturbance - overview:.png', 'Histogram.png', 'last.png'
    ]

    for fig_name in figures:
        plt.figure()  # Create a new figure to avoid overlap
        if fig_name == 'Plot_Mean_Conductance.png':
            for i in range(4):
                plt.subplot(2, 2, i + 1)
                plt.plot(mean_G_hist[i, :], linewidth=2)
                plt.axhline(y=target[i], color='r', linestyle='--')
                plt.title(f'Target = {target[i]} uS')
                plt.xlabel('Programm iter.')
                plt.ylabel('Mean conductance (uS)')
                plt.grid(True)
            plt.tight_layout()

        elif fig_name == 'Plot std conductance vs. Program iteration for each target.png':
            for i in range(4):
                plt.subplot(2, 2, i + 1)
                plt.plot(std_G_hist[i, :], linewidth=2)
                plt.title(f'Target = {target[i]} uS')
                plt.xlabel('Programm iter.')
                plt.ylabel('std. conductance (uS)')
                plt.grid(True)
            plt.tight_layout()

        elif fig_name == 'Plot mean Vwl vs. Program iteration for each target.png':
            for i in range(4):
                plt.subplot(2, 2, i + 1)
                plt.plot(mean_vwl_hist[i, :], linewidth=2)
                plt.title(f'Target = {target[i]} uS')
                plt.xlabel('Programm iter.')
                plt.ylabel('Mean Vwl (V)')
                plt.grid(True)
            plt.tight_layout()

        elif fig_name == 'Plot std Vwl vs. Program iteration for each target.png':
            for i in range(4):
                plt.subplot(2, 2, i + 1)
                plt.plot(std_vwl_hist[i, :], linewidth=2)
                plt.title(f'Target = {target[i]} uS')
                plt.xlabel('Programm iter.')
                plt.ylabel('std Vwl (V)')
                plt.grid(True)
            plt.tight_layout()

        elif fig_name == 'Check finetune disturbance - overview:.png':
            climit = [-20, 20]
            diff = read_after_finetune - np.squeeze(processed_read_hist[:, :, -1])
            plt.imshow(diff, cmap='viridis', vmin=climit[0], vmax=climit[1])
            plt.title('G diff = Read after finetune - last fine tune read')
            plt.colorbar()
            plt.clim(climit)

        elif fig_name == 'Histogram.png':
            diff = read_after_finetune - np.squeeze(processed_read_hist[:, :, -1])
            plt.hist(diff.ravel(), bins=100)
            mean_diff = np.mean(diff)
            std_diff = np.std(diff)
            plt.title(f'Mean = {mean_diff:.2f}; std = {std_diff:.2f}')

        elif fig_name == 'last.png':
            diff = read_after_finetune - np.squeeze(processed_read_hist[:, :, -1])
            read_outliner = diff > read_outliner_threshold
            read_outliner_per_WL = np.sum(read_outliner, axis=0)
            read_outliner_per_BL = np.sum(read_outliner, axis=1)
            plt.subplot(1, 2, 1)
            plt.scatter(np.arange(1, 1297), read_outliner_per_WL, marker='o')
            plt.xlabel('WL index')
            plt.ylabel(f'Read outliner number (> {read_outliner_threshold} uS)')
            plt.title('Outliers per WL')
            plt.subplot(1, 2, 2)
            plt.scatter(np.arange(1, 65), read_outliner_per_BL, marker='o')
            plt.xlabel('BL index')
            plt.ylabel(f'Read outliner number (> {read_outliner_threshold} uS)')
            plt.title('Outliers per BL')
            plt.tight_layout()

        # Save the plot to a BytesIO stream and encode it to base64
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        encoded_plots.append(plot_url)
        plt.close()

    return encoded_plots




