#auto_uplaod.py
#Lock the table (prevent it from shown or being utilized  on the database) before it is completely uploaded to the database, only unlock the file 9available to be shown on the database and can be utilized) after it completely uploaded to database
#endured error occurs during the processing of a file (such as the ValueError you encountered), it will be caught, and the script will continue to monitor and process other files.
import os
import time
import numpy as np
import pandas as pd
from sqlalchemy import inspect
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from urllib.parse import quote_plus
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading

class NPYHandler(FileSystemEventHandler):
    def __init__(self, engine):
        self.engine = engine

    def is_file_stable(self, file_path, wait_time=10, check_interval=2):
        """Check if the file size is stable for 'wait_time' seconds."""
        last_size = -1
        stable_time = 0

        while stable_time < wait_time:
            try:
                current_size = os.path.getsize(file_path)
            except FileNotFoundError:
                return False

            if current_size == last_size:
                stable_time += check_interval
            else:
                stable_time = 0
                last_size = current_size

            time.sleep(check_interval)

        return True

    def process(self, event):
        print(f"Processing event: {event}")  # Debug: Print the event details
        if event.is_directory:
            return

        if event.event_type in ('created') and event.src_path.endswith('.npy'):
            if self.is_file_stable(event.src_path):
                try:
                    data = np.load(event.src_path, allow_pickle=True)
                    if data.ndim == 1:
                        df = pd.DataFrame(data)
                    else:
                        print(f"Error: Data in file {event.src_path} is not 1-dimensional.")
                        return

                    table_name = os.path.basename(event.src_path).replace(".npy", "")

                    with self.engine.connect() as conn:
                        try:
                            # Lock the table before uploading
                            conn.execute(f"LOCK TABLES {table_name} WRITE")
                            df.to_sql(table_name, con=conn, if_exists='replace', index=False)
                            print(f"File {event.src_path} synced with local RDS")
                        except Exception as e:
                            print(f"Error during database operations for {table_name}: {e}")
                        finally:
                            # Unlock the tables after uploading is done
                            conn.execute("UNLOCK TABLES")

                except Exception as e:
                    print(f"Error processing file {event.src_path}: {e}")

    '''def on_created(self, event):
        print(f"File created: {event.src_path}")  # Debug: Print the created file path
        self.process(event)'''

    def on_created(self, event):
        threading.Thread(target=self.process, args=(event,)).start()
    
    #" Utilizes Python's threading module to handle file events in separate threads. This allows multiple files to be processed simultaneously, improving responsiveness and throughput when many files are being monitored."

def main():
    DB_HOST = "192.168.68.164"
    DB_USER = "root"
    MYSQL_PASSWORD_RAW = 'Str0ng_P@ssw0rd!'
    MYSQL_PASSWORD = quote_plus(MYSQL_PASSWORD_RAW)

    # Database connections
    engine = create_engine(f"mysql+mysqlconnector://{DB_USER}:{MYSQL_PASSWORD}@{DB_HOST}")

    try:
        # List databases
        inspector = inspect(engine)
        databases = inspector.get_schema_names()
        print("\nAvailable databases in local RDS:")
        for db in databases:
            print(db)

        # Database selection
        selected_db = input("\nPlease enter the database to upload to: ")
        if selected_db not in databases:
            print("Invalid database selected for local RDS.")
            return
        
        #selected_db = 'endurance'

        engine = create_engine(f"mysql+mysqlconnector://{DB_USER}:{MYSQL_PASSWORD}@{DB_HOST}/{selected_db}")

        # Directory monitoring
        #directory_to_watch = input("Please enter the directory to monitor: ")
        directory_to_watch = '/home/nuc13/new_test_env/npu_v1p_unified_test_env/BACKEND_driver/endurance_test_results'
        print(f"Monitoring directory: {directory_to_watch}")  # Debug: Confirm the directory being monitored
        event_handler = NPYHandler(engine)
        observer = Observer()
        observer.schedule(event_handler, path=directory_to_watch, recursive=False)
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
