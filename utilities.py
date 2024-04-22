import re
from flask import render_template_string, request
import pandas as pd
from scipy.io import loadmat
import h5py
import numpy as np
from io import BytesIO
import scipy.io

def sanitize_table_name(name):
    """
    Sanitize the filename to make it suitable for usage as a MySQL table name.
    """
    # Remove non-word characters and spaces
    sanitized_name = re.sub(r'\W+| ', '_', name)

    # Ensure it starts with a letter, prepend an 'a' if not
    #if not sanitized_name[0].isalpha():
        #sanitized_name = 'a' + sanitized_name

    return sanitized_name.lower()

def validate_filename(filename):
    print("Filename:", filename)
    #pattern = r"^lot[A-Za-z0-9]+_wafer[A-Za-z0-9]+_die[A-Za-z0-9]+_dut[A-Za-z0-9]+_[A-Za-z0-9]+_[A-Za-z0-9]+_[A-Za-z0-9]+_[A-Za-z0-9]+\.(csv|txt)$"
    pattern = r"^lot[A-Za-z0-9]+_wafer[A-Za-z0-9]+_die[A-Za-z0-9]+_dut[A-Za-z0-9]*_[A-Za-z0-9]+_[A-Za-z0-9]+_[A-Za-z0-9]+_[A-Za-z0-9]+\.(csv|txt|npy)$"
    print("Pattern match result:", bool(re.match(pattern, filename)))
    return bool(re.match(pattern, filename))

def render_results(results):
    results_html = "<html><body style='background-color: white;'>"
    for result in results:
        results_html += f"<p>{result}</p>"
    results_html += "</body></html>"
    return render_template_string(results_html), 200

'''def get_form_data_generate_plot_VCR(form):
    form_data = {key: request.form.get(key, "").strip() for key in ['selected_groups', 'selected_setting', 'num_settings']}

    # Convert selected_groups to a list of integers
    selected_groups = form_data['selected_groups'].split(',') if form_data['selected_groups'] else []
    form_data['selected_groups'] = [int(float(num)) for num in selected_groups]

    selected_setting = form_data['selected_setting'].split(',') if form_data['selected_setting'] else []
    form_data['selected_setting'] = [int(float(num)) for num in selected_setting]

    form_data['num_settings'] = float(form_data['num_settings']) if form_data['num_settings'] else None

    return form_data'''

'''def get_form_data_generate_plot(form):  # Accept a 'form' parameter
    #form_data = {key: form.get(key, "").strip() for key in ['selected_groups']}
    form_data = {key: request.form.get(key, "").strip() for key in ['selected_groups', 'sub_array_size']}

    # Convert selected_groups to a list of integers
    selected_groups = form_data['selected_groups'].split(',') if form_data['selected_groups'] else []
    form_data['selected_groups'] = [int(float(num)) for num in selected_groups]

    form_data['sub_array_size'] = float(form_data['sub_array_size']) if form_data['sub_array_size'] else None

    return form_data'''

def get_form_data_generate_plot(form):
    # Extract form data for selected_groups and sub_array_size
    form_data = {key: request.form.get(key, "").strip() for key in ['selected_groups', 'sub_array_size']}

    # Convert selected_groups to a list of integers
    selected_groups = form_data['selected_groups'].split(',') if form_data['selected_groups'] else []
    form_data['selected_groups'] = [int(float(num)) for num in selected_groups]

    # Split sub_array_size into two integers
    sub_array_size = form_data['sub_array_size'].split(',') if form_data['sub_array_size'] else []
    form_data['sub_array_size'] = [int(float(num)) for num in sub_array_size] if len(sub_array_size) == 2 else [None, None]

    return form_data

'''def get_form_data_generate_plot_TCR_separate(form):  # Accept a 'form' parameter
    #form_data = {key: form.get(key, "").strip() for key in ['selected_groups']}
    form_data = {key: request.form.get(key, "").strip() for key in ['selected_groups', 'num_settings']}

    # Convert selected_groups to a list of integers
    selected_groups = form_data['selected_groups'].split(',') if form_data['selected_groups'] else []
    form_data['selected_groups'] = [int(float(num)) for num in selected_groups]

    form_data['num_settings'] = float(form_data['num_settings']) if form_data['num_settings'] else None

    return form_data'''

'''def get_form_data_generate_plot_64x64(form):  # Accept a 'form' parameter
    # Extract the 'date_option' field from the form
    form_data = {
        'date_option': form.get('date_option', "").strip()
    }

    return form_data'''

'''def get_form_data_endurance(form):
    # Extract values from form, defaulting to an empty string and then stripping whitespace.
    form_data = {key: form.get(key, "").strip() for key in ['side_of_a_square', 'side_subsquare', 'section_number', 'selected_num_set']}

    # Convert 'side_of_a_square' to integer if it's provided, otherwise set to None.
    form_data['side_of_a_square'] = int(float(form_data['side_of_a_square'])) if form_data['side_of_a_square'] else None

    # Convert 'side_subsquare' to integer if it's provided, otherwise set to None.
    form_data['side_subsquare'] = int(float(form_data['side_subsquare'])) if form_data['side_subsquare'] else None

    # Convert 'section_number' to a list of integers, splitting by comma if provided, otherwise an empty list.
    section_numbers = form_data['section_number'].split(',') if form_data['section_number'] else []
    form_data['section_numbers'] = [int(num) for num in section_numbers]

    # Convert 'selected_num_set' to a list of integers, splitting by comma if provided, otherwise an empty list.
    selected_num_set = form_data['selected_num_set'].split(',') if form_data['selected_num_set'] else []
    form_data['selected_num_set'] = [int(num) for num in selected_num_set]

    return form_data'''

'''def get_form_data_generate_plot():
    form_data = {key: request.form.get(key, "").strip() for key in ['selected_groups']}

    selected_groups = form_data['selected_groups'].split(',') if form_data['selected_groups'] else []
    form_data['selected_groups'] = [float(num) for num in selected_groups]

    return form_data'''

'''def get_form_data(form):
    form_data = {key: request.form.get(key, "").strip() for key in ['side_of_a_square', 'side_subsquare', 'section_number']}

    # Convert to int if values are provided and are valid numbers
    form_data['side_of_a_square'] = int(float(form_data['side_of_a_square'])) if form_data['side_of_a_square'] else None
    form_data['side_subsquare'] = int(float(form_data['side_subsquare'])) if form_data['side_subsquare'] else None
    form_data['section_number'] = int(float(form_data['section_number'])) if form_data['section_number'] else None

    return form_data'''

'''def get_form_data():
    form_data = {key: request.form.get(key, "").strip() for key in ['side_of_a_square', 'side_subsquare']}
    form_data['side_of_a_square'] = float(form_data['side_of_a_square']) if form_data['side_of_a_square'] else 128
    form_data['side_subsquare'] = float(form_data['side_subsquare']) if form_data['side_subsquare'] else 32

    return form_data'''

'''
def get_form_data():
    form_data = {key: request.form.get(key, "").strip() for key in ['start_value', 'end_value', 'left_y_axis_label', 'right_y_axis_label', 'x_axis_label', 'side_of_a_square']}

    form_data['start_value'] = float(form_data['start_value']) if form_data['start_value'] else None
    form_data['end_value'] = float(form_data['end_value']) if form_data['end_value'] else None
    form_data['side_of_a_square'] = float(form_data['side_of_a_square']) if form_data['side_of_a_square'] else 128

    return form_data'''

'''def get_form_data():
    form_data = {key: request.form.get(key, "").strip() for key in ['start_value', 'end_value', 'left_y_axis_label', 'right_y_axis_label', 'x_axis_label', 'side_of_a_square']}

    try:
        form_data['start_value'] = float(form_data['start_value'])
    except ValueError:
        form_data['start_value'] = None

    try:
        form_data['end_value'] = float(form_data['end_value'])
    except ValueError:
        form_data['end_value'] = None

    try:
        form_data['side_of_a_square'] = float(form_data['side_of_a_square'])
    except ValueError:
        form_data['side_of_a_square'] = 128

    return form_data'''

def flatten_sections(array_3d):
    """Flatten 16x16 sections from a 3D array."""
    slices = [array_3d[row:row+16, col:col+16, setting].flatten()
              for setting in range(array_3d.shape[2])
              for row in range(0, 64, 16)
              for col in range(0, 64, 16)]
    return np.concatenate(slices)

def process_mat_file(file_content):
    df = None
    mat_data = scipy.io.loadmat(BytesIO(file_content))
    for key in mat_data:
        if not key.startswith('__'):
            data = mat_data[key]
            print("data.shape", data.shape)
            if data.shape == (64, 64, 8) or data.shape == (64, 64, 4) or data.shape == (100, 64, 64):
                flattened_data = flatten_sections(data)
                df = pd.DataFrame(flattened_data)
            elif data.shape == (64, 64):  # Handling 2D data
                df = pd.DataFrame(data)
            if df is not None:
                break
    return df

def process_h5py_file(file_stream):  #64x64
    df = None
    with h5py.File(file_stream, 'r') as f:
        for key in f.keys():
            data = f[key]
            if isinstance(data, h5py.Dataset) and data.shape == (64, 64):
                df = pd.DataFrame(data[:]).T
                break
    return df

def is_hdf5_file(file_stream):
    # Check if the file stream is an HDF5 file
    try:
        h5py.File(file_stream, 'r')
        return True
    except OSError:
        return False

def process_file(file_stream, file_extension, db_name):
    file_stream.seek(0)
    df = None

    if file_extension == "csv":
        options = {'checkerboard': (None, None), 'forming_voltage_map': (None, 4)}
        header_option, skip_rows_option = options.get(db_name, (0, None))
        df = pd.read_csv(file_stream, header=header_option, skiprows=skip_rows_option)

    elif file_extension == "txt":
        content = file_stream.read()
        if len(content) > 65535:
            raise ValueError("Content too large to fit in the database column")
        df = pd.DataFrame({'content': [content]})

    elif file_extension == "npy":
        try:
            # Load the .npy file into a numpy array
            np_array = np.load(file_stream, allow_pickle=True)
            # Convert the numpy array to a pandas DataFrame
            if np_array.ndim == 1:
                df = pd.DataFrame(np_array)
            elif np_array.ndim == 2:
                # For 2-dimensional arrays, we use the whole array to create the DataFrame
                df = pd.DataFrame(data=np_array)
            else:
                raise ValueError("Numpy array is not 1-dimensional or 2-dimensional")
        except Exception as e:
            print(f"Error processing .npy file: {e}")
            df = pd.DataFrame()

    elif file_extension == "mat":
        # Reset the file stream to the beginning for reading
        file_stream.seek(0)
        if is_hdf5_file(file_stream):
            # If it's an HDF5 file, use the h5py processor
            df = process_h5py_file(file_stream)
        else:
            # For other .mat files, process them here
            try:
                # Reset the file stream again as is_hdf5_file may have moved it
                file_stream.seek(0)
                file_content = file_stream.read()
                df = process_mat_file(file_content)
                if df is None or df.empty:
                    raise ValueError("No suitable dataset found in the .mat file")
            except Exception as e:
                print(f"Error processing .mat file: {e}")
                df = pd.DataFrame()

    # Fallback for any unprocessed or empty data frames
    if df is None or df.empty:
        print("No data processed for the file, returning empty DataFrame")
        df = pd.DataFrame()

    return df
