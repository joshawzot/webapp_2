import numpy as np
import scipy.io as sio
import os
import re
from datetime import datetime

# Function to extract datetime from filename
def extract_datetime(filename):
    pattern = r"(\d{4}-\d{2}-\d{2}_\d{6})"
    match = re.search(pattern, filename)
    if match:
        return datetime.strptime(match.group(1), "%Y-%m-%d_%H%M%S")
    return None

# General function to combine .mat files
def combine_mat_files(x, y, directory=os.getcwd()):
    # List all .mat files in the directory
    mat_files = [f for f in os.listdir(directory) if f.endswith('.mat')]
    
    # Sort files by datetime extracted from filenames
    mat_files.sort(key=lambda f: extract_datetime(f))

    # Check if the number of files matches x*y
    if len(mat_files) != x * y:
        raise ValueError(f"Number of .mat files ({len(mat_files)}) does not match expected count ({x * y}).")

    # Initialize variables to determine the size of the final array
    first_file_data = sio.loadmat(os.path.join(directory, mat_files[0]))
    first_key = next(key for key in first_file_data.keys() if not key.startswith('__'))
    first_data = first_file_data[first_key]
    
    # Squeeze the data if it is a 3D array with a singleton dimension
    if first_data.ndim == 3 and first_data.shape[0] == 1:
        first_data = np.squeeze(first_data)
    
    m, n = first_data.shape

    # Initialize the final array
    final_array = np.zeros((x * m, y * n))

    # Populate the final array in the specified order
    for idx, file in enumerate(mat_files):
        # Load the current .mat file
        mat_data = sio.loadmat(os.path.join(directory, file))
        key = next(key for key in mat_data.keys() if not key.startswith('__'))
        data = mat_data[key]
        
        # Squeeze the data if it is a 3D array with a singleton dimension
        if data.ndim == 3 and data.shape[0] == 1:
            data = np.squeeze(data)
        
        if data.shape != (m, n):
            raise ValueError(f"File {file} does not contain an array of shape ({m}, {n}).")

        # Calculate the position in the final array
        row_section = (idx % x) * m
        col_section = (idx // x) * n

        # Place the data in the final array
        final_array[row_section:row_section + m, col_section:col_section + n] = data

    # Save the final array to a new .mat file
    sio.savemat('combined.mat', {'combined_data': final_array})

    print('Combined .mat file created successfully.')

# Example usage
combine_mat_files(4, 10)