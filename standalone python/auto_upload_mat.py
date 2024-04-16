# python3 auto_upload_mat.py

#before running the script, install packages  #ubuntu
'''sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install python3-pip -y
sudo apt-get install libhdf5-dev -y
sudo apt-get install libpq-dev -y
sudo apt-get install default-libmysqlclient-dev -y
pip3 install numpy pandas scipy h5py sqlalchemy watchdog'''

#windows
'''python -m pip install --upgrade pip
pip install numpy pandas scipy h5py sqlalchemy watchdog'''

import os
import re
import time
import numpy as np
import pandas as pd
import scipy.io
import h5py
from io import BytesIO
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import SQLAlchemyError
from urllib.parse import quote_plus
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from sqlalchemy.exc import SQLAlchemyError
import threading

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
            if data.shape == (64, 64, 8):
                flattened_data = flatten_sections(data)
                df = pd.DataFrame(flattened_data)
            elif data.shape == (64, 64):  # Handling 2D data
                df = pd.DataFrame(data)
            if df is not None:
                break
    return df

def process_h5py_file(file_stream):
    df = None
    with h5py.File(file_stream, 'r') as f:
        for key in f.keys():
            data = f[key]
            if isinstance(data, h5py.Dataset) and data.shape == (64, 64):
                df = pd.DataFrame(data[:]).T
                break
    return df

def is_hdf5_file(file_stream):
    try:
        h5py.File(file_stream, 'r')
        return True
    except OSError:
        return False

class NPYHandler(FileSystemEventHandler):
    def __init__(self, engine):
        self.engine = engine

    def process(self, event):
        print(f"Processing event: {event}")
        if event.is_directory:
            return
 
        file_path = event.src_path
        print('file_path:', file_path)
        file_path_sanitized = sanitize_table_name(file_path)
        print('file_path_sanitized:', file_path_sanitized)
        file_extension = file_path_sanitized.rpartition('_')[-1]
        print('file_extension:', file_extension)

        if event.event_type in ('created'):
            try:
                if file_extension == 'mat':
                    with open(file_path, 'rb') as file_stream:
                        if is_hdf5_file(file_stream):
                            df = process_h5py_file(file_stream)
                        else:
                            file_stream.seek(0)
                            file_content = file_stream.read()
                            df = process_mat_file(file_content)

                        if df is None or df.empty:
                            raise ValueError("No suitable data found in .mat file")
                else:
                    print(f"Unsupported file type: {file_extension}")
                    return

                # Extract the basename, sanitize it, and then remove the actual file extension
                sanitized_basename = sanitize_table_name(os.path.basename(file_path))
                table_name, _ = os.path.splitext(sanitized_basename)  # Correctly removing the extension
                print('table_name:', table_name)
                with self.engine.connect() as conn:
                    df.to_sql(table_name, con=conn, if_exists='replace', index=False)
                    print(f"File {file_path} synced RDS")

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

    '''def on_created(self, event):
        print(f"File created: {event.src_path}")
        self.process(event)'''

    def on_created(self, event):
        print(f"File created: {event.src_path}")
        # Start a new thread for processing the event
        processing_thread = threading.Thread(target=self.process, args=(event,))
        processing_thread.start()

def main():
    '''DB_HOST = "192.168.68.164"
    DB_USER = "root"
    MYSQL_PASSWORD_RAW = 'Str0ng_P@ssw0rd!'
    MYSQL_PASSWORD = quote_plus(MYSQL_PASSWORD_RAW)
    engine = create_engine(f"mysql+mysqlconnector://{DB_USER}:{MYSQL_PASSWORD}@{DB_HOST}")    
    '''

    DB_HOST = "webapp.cdecmst6qwog.us-east-2.rds.amazonaws.com"
    DB_USER = "admin"
    MYSQL_PASSWORD_RAW = 'Aa11720151015'
    MYSQL_PASSWORD = quote_plus(MYSQL_PASSWORD_RAW)
    DB_PORT = 3306  # Added port number for the database connection
    engine = create_engine(f"mysql+mysqlconnector://{DB_USER}:{MYSQL_PASSWORD}@{DB_HOST}:{DB_PORT}")

    try:
        inspector = inspect(engine)
        databases = inspector.get_schema_names()
        print("\nAvailable databases:")
        for db in databases:
            print(db)

        selected_db = input("\nPlease enter the database to upload to: ")
        if selected_db not in databases:
            print("Invalid database selected.")
            return
        
        #Local DB
        #engine = create_engine(f"mysql+mysqlconnector://{DB_USER}:{MYSQL_PASSWORD}@{DB_HOST}/{selected_db}")

        #AWS RDS
        engine = create_engine(f"mysql+mysqlconnector://{DB_USER}:{MYSQL_PASSWORD}@{DB_HOST}:{DB_PORT}/{selected_db}")

        # Prompt the user to enter the directory to monitor
        directory_to_watch = input("Please enter the directory to monitor: ")
        if not os.path.exists(directory_to_watch):
            print(f"Directory does not exist: {directory_to_watch}")
            return

        print(f"Monitoring directory: {directory_to_watch}")
        event_handler = NPYHandler(engine)
        observer = Observer()
        observer.schedule(event_handler, path=directory_to_watch, recursive=True)
        observer.start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()

    except SQLAlchemyError as e:
        print(f"Database error: {e}")

if __name__ == "__main__":
    main()