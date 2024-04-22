#route_handlers.py

# Standard library imports
import re
from io import BytesIO
import time

# External libraries
import pandas as pd
import mysql.connector
from flask import (Flask, render_template, redirect, url_for, request, session, flash, jsonify, render_template_string, send_file)
from werkzeug.security import check_password_hash, generate_password_hash

import hashlib
import boto3
from pptx import Presentation
from reportlab.pdfgen import canvas
import os
import tempfile
import boto3
import requests
# Flask extensions
from flask import Flask, request, redirect, url_for
from threading import Thread
from urllib.parse import quote_plus
from flask import request, send_file
import base64
from pptx.util import Inches
from flask import Flask, request, make_response

#------------------------------------------------------------------------
# Custom module imports
from my_flask_app import app, cache
from db_operations import create_connection, fetch_data, close_connection, create_db_engine, create_db, get_all_databases, connect_to_db, fetch_tables, rename_database, move_tables, copy_tables, copy_all_tables, copy_tables_2, move_tables_2
from utilities import sanitize_table_name, validate_filename, render_results, get_form_data_generate_plot, process_file
#from generate_plot_vertical_xn import generate_plot_vertical_xn
#from generate_plot_horizontal_boxplotsigma_xn import generate_plot_horizontal_boxplotsigma_xn
#from generate_plot_horizontal_boxplotsigma_xnxm import generate_plot_horizontal_boxplotsigma_xnxm
#from generate_plot_horizontal_sigma_xnxm import generate_plot_horizontal_sigma_xnxm
#from generate_plot_forming_voltage_map import generate_plot_forming_voltage_map
#from generate_plot_percentage_bars import generate_plot_percentage_bars
#from generate_plot_ttc import generate_plot_ttc
from generate_plot_endurance import generate_plot_endurance
#from generate_plot_checkerboard import generate_plot_checkerboard
from generate_plot import generate_plot
from generate_plot_combined import generate_plot_combined
from generate_plot_separate import generate_plot_separate
#from generate_plot_64x64 import generate_plot_64x64
#from generate_plot_VCR import generate_plot_VCR
#from generate_plot_TCR import generate_plot_TCR
#from generate_plot_TCR_separate import generate_plot_TCR_separate
#from generate_plot_cycling import generate_plot_cycling

'''from flask_caching import Cache
cache = Cache(app, config={'CACHE_TYPE': 'simple'})'''

import subprocess
from flask import send_file

#backup to server side
'''
@app.route('/backup', methods=['POST'])
def backup():
    data = request.get_json()
    local_backup_dir = data.get('backupDirectory')
    print("local_backup_dir:", local_backup_dir)
    dbnames = data.get('databaseNames')
    print("dbnames:", dbnames)

    if not local_backup_dir or not dbnames:
        return jsonify({"error": "No backup directory or databases provided"}), 400

    # Ensure the backup directory exists
    print("os.path.exists(local_backup_dir):", os.path.exists(local_backup_dir))
    if not os.path.exists(local_backup_dir):
        try:
            os.makedirs(local_backup_dir)
        except OSError as e:
            print("fucked")
            return jsonify({"error": f"Failed to create backup directory: {e}"}), 500

    responses = []
    for dbname in dbnames:
        backup_file = f"{dbname}_{int(time.time())}.sql"
        backup_path = os.path.join(local_backup_dir, backup_file)

        command = f"mysqldump -u{DB_CONFIG['DB_USER']} -p'{DB_CONFIG['MYSQL_PASSWORD_RAW']}' {dbname} > {backup_path}"
        try:
            result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
            responses.append({"database": dbname, "message": f"Backup created successfully at {backup_path}"})
        except subprocess.CalledProcessError as e:
            error_message = e.stderr
            responses.append({"database": dbname, "error": f"Failed to create backup: {error_message}"})

    return jsonify(responses), 200
'''

import zipfile
#back up to client side
@app.route('/backup', methods=['POST'])
def backup():
    data = request.get_json()
    dbnames = data.get('databaseNames')

    if not dbnames:
        return jsonify({"error": "No databases provided"}), 400

    local_backup_dir = '/home/ubuntu/backup'
    os.makedirs(local_backup_dir, exist_ok=True)

    responses = []
    zip_filename = f"backups_{int(time.time())}.zip"
    zip_path = os.path.join(local_backup_dir, zip_filename)

    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for dbname in dbnames:
            backup_file = f"{dbname}_{int(time.time())}.sql"
            backup_path = os.path.join(local_backup_dir, backup_file)

            command = f"mysqldump -h {DB_CONFIG['DB_HOST']} -P {DB_CONFIG['RDS_PORT']} -u{DB_CONFIG['DB_USER']} -p'{DB_CONFIG['MYSQL_PASSWORD_RAW']}' {dbname} > {backup_path}"
            try:
                subprocess.run(command, shell=True, check=True)
                zipf.write(backup_path, arcname=backup_file)
                os.remove(backup_path)  # Remove the SQL file after adding to zip
                responses.append({"database": dbname, "message": f"Backup created successfully at {backup_path}"})
            except subprocess.CalledProcessError as e:
                error_message = e.stderr
                responses.append({"database": dbname, "error": f"Failed to create backup: {error_message}"})

    # After creating zip file, send it to the client
    print("local_backup_dir:", local_backup_dir)
    print("zip_filename:", zip_filename)
    return send_from_directory(local_backup_dir, zip_filename, as_attachment=True)

@app.route('/all-databases')
def all_databases():
    conn = create_connection()
    cursor = conn.cursor()
    databases = get_all_databases(cursor)  # Fetch all database names
    cursor.close()
    conn.close()

    if ENG:
        return render_template('list_databases_eng.html', databases=databases)
    else:
        return render_template('list_databases.html', databases=databases)

from config import DB_CONFIG, LOCAL_DB, ENG
@app.route('/delete-database/<name>', methods=['DELETE'])
def delete_database(name):
    conn = connect_to_db(DB_CONFIG['DB_USER'], DB_CONFIG['MYSQL_PASSWORD_RAW'], DB_CONFIG['DB_HOST'], DB_CONFIG['RDS_PORT'] if not LOCAL_DB else None)
    if conn is None:
        return jsonify({"error": "Failed to connect to the database"}), 500

    try:
        cursor = conn.cursor()
        cursor.execute(f"DROP DATABASE IF EXISTS {name}")
        conn.commit()
        return jsonify({"message": f"Database {name} deleted successfully."}), 200
    except mysql.connector.Error as err:
        return jsonify({"error": f"Failed to delete database: {err}"}), 500
    finally:
        conn.close()

@app.route('/')  # Defines a route for the root URL
def home():
    return redirect(url_for('home_page'))

@app.route('/save-txt-content/<database>/<table_name>', methods=['POST'])
def save_txt_content(database, table_name):
    try:
        content = request.json['content']
        connection = create_connection(database)
        cursor = connection.cursor()
        query = f"UPDATE `{table_name}` SET content = %s WHERE content IS NOT NULL LIMIT 1"
        cursor.execute(query, (content,))
        connection.commit()
        close_connection()
        return "Content saved successfully", 200
    except mysql.connector.Error as err:
        return str(err), 400

import os
import hashlib
from flask import Flask, send_from_directory

last_pptx_hash = ''  # Just a starting value; should be managed appropriately.
@app.route('/get-pptx-as-pdf', methods=['GET'])
def get_pptx_as_pdf():
    global last_pptx_hash

    pptx_path = os.path.join('static', 'database.pptx')
    pdf_path = os.path.join('static', 'database.pdf')

    # Check if PPTX has changed
    with open(pptx_path, 'rb') as f:
        current_hash = hashlib.md5(f.read()).hexdigest()

    if current_hash != last_pptx_hash:
        # Convert PPTX to PDF using LibreOffice
        cmd = f'libreoffice --headless --convert-to pdf --outdir {os.path.dirname(pdf_path)} {pptx_path}'
        os.system(cmd)

        # Update the hash value
        last_pptx_hash = current_hash

    return send_from_directory('static', 'database.pdf')

'''
# AWS S3 Configuration
s3_client = boto3.client('s3', aws_access_key_id='AKIAU62NNZW42NLXZZ6B', aws_secret_access_key='9ZvgGiZi8KK4Glb7S/4CdFx4sVuFrfK6ySY82K9YY')
bucket_name = 'webappcdn'

# Global variable to track the last hash
last_pptx_hash = ''
@app.route('/get-pptx-as-pdf', methods=['GET'])
def get_pptx_as_pdf(): hash
    conn = sqlite3.connect('hash_store.db')
    c = conn.cursor()
    c.execute('SELECT hash FROM last_hash WHERE id = 1')
    last_pptx_hash, = c.fetchone()

    if current_hash != last_pptx_hash:
        # Process to convert PPTX to PDF (e.g., using LibreOffice)
        # This is a placeholder. Implement the conversion logic
        with tempfile.NamedTemporaryFile(suffix=".pptx", delete=False) as tmp_pptx:
            tmp_pptx.write(pptx_response.content)
            tmp_pptx_path = tmp_pptx.name
        pdf_path = tmp_pptx_path.replace('.pptx', '.pdf')
        cmd = f'libreoffice --headless --convert-to pdf --outdir {os.path.dirname(pdf_path)} {tmp_pptx_path}'
        os.system(cmd)

        # Upload the PDF back to S3
        with open(pdf_path, 'rb') as pdf_file:
            s3_client.put_object(Bucket=bucket_name, Key='database.pdf', Body=pdf_file)

        # Cleanup temporary files
        os.remove(tmp_pptx_path)
        os.remove(pdf_path)

        # Update the hash value in the database
        c.execute('UPDATE last_hash SET hash = ? WHERE id = 1', (current_hash,))
        conn.commit()

    conn.close()

    # Fetch and return the updated PDF from S3
    pdf_url = "https://d2lsypbr1lit3e.cloudfront.net/database.pdf"
    pdf_response = requests.get(pdf_url)
    return send_file(io.BytesIO(pdf_response.content), attachment_filename='database.pdf')
'''

@app.route('/home-page', methods=['GET'])
def home_page():
    """Get a list of databases available."""
    try:
        conn = create_connection()
        cursor = conn.cursor()
        databases = get_all_databases(cursor)  # Fetch all database names
        cursor.close()
        conn.close()

        # Creating a dictionary to map database names to image URLs
        '''image_urls = {
            "vertical_xn": url_for('static', filename='example_plot_vertical_xn.png'),
            "horizontal_boxplotsigma_xn": url_for('static', filename='example_plot_horizontal_boxplotsigma_xn.png'),
            "horizontal_sigma_xnxm": url_for('static', filename='example_plot_horizontal_sigma_xnxm.png'),
            "forming_voltage_map": url_for('static', filename='example_plot_forming_voltage_map.png'),
            "percentage_bars": url_for('static', filename='example_plot_percentage_bars.png'),
            "ttc": url_for('static', filename='example_plot_ttc.png'),
            "endurance": url_for('static', filename='example_plot_endurance.png'),
            "horizontal_boxplotsigma_xnxm": url_for('static', filename='example_plot_horizontal_boxplotsigma_xnxm.png'),
            "checkerboard": url_for('static', filename='example_plot_checkerboard.png')
        }
        #powerpoint_url = url_for('static', filename='database.pdf')
        powerpoint_url = url_for('static', filename='database.pptx')'''

        '''cloudfront_base_url = "https://d2lsypbr1lit3e.cloudfront.net/"
        image_urls = {
            "vertical_xn": cloudfront_base_url + "static/example_plot_vertical_xn.png",
            "horizontal_boxplotsigma_xn": cloudfront_base_url + "static/example_plot_horizontal_boxplotsigma_xn.png",
            "horizontal_sigma_xnxm": cloudfront_base_url + "static/example_plot_horizontal_sigma_xnxm.png",
            "forming_voltage_map": cloudfront_base_url + "static/example_plot_forming_voltage_map.png",
            "percentage_bars": cloudfront_base_url + "static/example_plot_percentage_bars.png",
            "ttc": cloudfront_base_url + "static/example_plot_ttc.png",
            "endurance": cloudfront_base_url + "static/example_plot_endurance.png",
            "endurance_multi": cloudfront_base_url + "static/example_plot_endurance.png",
            "horizontal_boxplotsigma_xnxm": cloudfront_base_url + "static/example_plot_horizontal_boxplotsigma_xnxm.png",
            "checkerboard": cloudfront_base_url + "static/example_plot_checkerboard.png"
        }'''

        #powerpoint_url = "https://d2lsypbr1lit3e.cloudfront.net/database.pptx"

        #return render_template('home_page.html', databases=databases, image_urls=image_urls, powerpoint_url=powerpoint_url)
        #return render_template('home_page.html', databases=databases, powerpoint_url=powerpoint_url)
        return render_template('home_page.html', databases=databases)
    except mysql.connector.Error as err:
        return str(err)

@app.route('/data-analysis', methods=['POST'])
def data_analysis():
    # Extract database and plot function from form and store in session
    session['database'] = request.form['database']
    database = session['database']
    plot_function = "None"

    try:
        tables = fetch_tables(database)  # Retrieve table data from the database

        # Prepare a comma-separated string of table names
        table_names = ','.join(table['table_name'] for table in tables)

        # Pass the necessary variables to the template
        #return render_template('list_tables.html', tables=tables, table_names=table_names, database=database)
        return render_template('list_tables.html', tables=tables, table_names=table_names, database=database, plot_function=plot_function)

    except mysql.connector.Error as err:
        # Log and handle any database errors gracefully
        print("MySQL Error: ", err)
        return str(err), 500

@app.route('/view-table/<database>/<table_name>', methods=['GET'])
def view_table(database, table_name):
    """View the content of a specific table."""
    print('database:', database)
    print('table_name:', table_name)
    try:
        connection = create_connection(database)
        cursor = connection.cursor()
        if table_name.endswith('_txt'):
            query = f"SELECT content FROM `{table_name}` LIMIT 1"
            results = fetch_data(cursor, query)
            close_connection()
            content = results[0][0] if results else ''
            print('------')
            print(table_name)
            return render_template('table_txt.html', database=database, content=content, table_name=table_name)
        else:
            query = f"SELECT * FROM `{table_name}`"
            results = fetch_data(cursor, query)
            column_names = [desc[0] for desc in cursor.description]
            close_connection()
            return render_template('table.html', results=results, column_names=column_names)
    except mysql.connector.Error as err:
        return str(err)

# Define a dictionary to map database names to functions
generate_plot_functions = {
    #"generate_plot_checkerboard": generate_plot_checkerboard,
    "generate_plot_endurance": generate_plot_endurance,
    #"generate_plot_ttc": generate_plot_ttc,
    #"generate_plot_percentage_bars": generate_plot_percentage_bars,
    #"generate_plot_vertical_xn": generate_plot_vertical_xn,
    #"generate_plot_horizontal_boxplotsigma_xn": generate_plot_horizontal_boxplotsigma_xn,
    #"generate_plot_horizontal_boxplotsigma_xnxm": generate_plot_horizontal_boxplotsigma_xnxm,
    #"generate_plot_horizontal_sigma_xnxm": generate_plot_horizontal_sigma_xnxm,
    #"generate_plot_forming_voltage_map": generate_plot_forming_voltage_map,
    "generate_plot": generate_plot,
    "generate_plot_combined": generate_plot_combined,
    "generate_plot_separate": generate_plot_separate,
    #"generate_plot_64x64": generate_plot_64x64,
    #"generate_plot_TCR": generate_plot_TCR,
    #"generate_plot_VCR": generate_plot_VCR,
    #"generate_plot_TCR_separate": generate_plot_TCR_separate,
    #"generate_plot_cycling": generate_plot_cycling,
}

@app.route('/render-plot/<unique_id>')
def render_plot(unique_id):
    # Attempt to fetch cached plot data using unique_id as the cache key
    cache_key = f"plot_data_{unique_id}"
    cached_plot_data = cache.get(cache_key)

    if cached_plot_data:
        # If cached data is found, use it to render the plot directly
        return render_template('plot.html', plot_data=cached_plot_data)

    # If no cache is found, retrieve the stored data from Redis
    stored_data_json = redis_client.get(unique_id)
    if not stored_data_json:
        return "Error: Invalid ID or Data Expired", 404
    
    stored_data = json.loads(stored_data_json)

    database = stored_data["database"]
    table_name = stored_data["table_name"]
    plot_function = stored_data["plot_function"]
    form_data = stored_data["form_data"]

    # Assuming generate_plot_functions is a dictionary mapping plot function names to their implementations
    generate_plot_function = generate_plot_functions.get(plot_function)
    if generate_plot_function is None:
        return "Error: Invalid plot function selection", 400

    try:
        plot_data = generate_plot_function(table_name.split(','), database, form_data=form_data)
        if not isinstance(plot_data, list):
            plot_data = [plot_data]

        # Cache the generated plot data for future requests
        cache.set(cache_key, plot_data, timeout=None)

        return render_template('plot.html', plot_data=plot_data)
    except Exception as e:
        return f"Error: {e}", 500

def fetch_all_numeric_data(table_name, database_name):
    # Establish a connection to the database
    connection = create_connection(database_name)
    data_dimension = "Unknown"
    try:
        with connection.cursor() as cursor:
            query = f"SELECT * FROM `{table_name}`"
            # Execute the query and fetch data
            cursor.execute(query)
            data = cursor.fetchall()
            # Determine the dimension of the data
            if data:
                # Check if the data has only one column (1D)
                if len(data[0]) == 1:
                    data_dimension = "1D"
                # If data has more than one column, consider it 2D
                else:
                    data_dimension = "2D"
                # If your application has a specific interpretation of 3D data,
                # you can add another condition here to set data_dimension to "3D"
    finally:
        connection.close()

    return data, data_dimension

import urllib.parse
from flask import Flask, request, jsonify, render_template, redirect
import uuid
import random
import numpy as np 
from my_flask_app import redis_client
import json

@app.route('/view-plot/<database>/<table_name>/<plot_function>', methods=['GET', 'POST'])
def view_plot(database, table_name, plot_function):

    if request.method == "POST":
        print("POST:::::::::::::::::::::::::::::::::")
        # Check if the user has made a plot function choice
        plot_function_choice = request.form.get('plot_choice')
        if plot_function_choice:
            plot_function = plot_function_choice
            if plot_function == "generate_plot" or "generate_plot_combined" or "generate_plot_separate":
                return render_template(f'input_form_generate_plot.html', database=database, table_name=table_name, plot_function=plot_function)             
            else:
                return jsonify({"error": "choice not selected"}), 400

        # Ensure plot_function has a value before proceeding
        if plot_function:
            print(f"plot_function: {plot_function}")  # Now plot_function should have a value
            
            # Conditional logic to handle form data based on plot_function
            if plot_function in ["generate_plot", "generate_plot_combined", "generate_plot_separate"]:
                form_data_handlers = {
                    "generate_plot": get_form_data_generate_plot,
                    "generate_plot_combined": get_form_data_generate_plot,
                    "generate_plot_separate": get_form_data_generate_plot,
                }
                form_data = form_data_handlers[plot_function](request.form)

                # Store the form data with a unique identifier in Redis
                unique_id = str(uuid.uuid4())
                redis_client.set(unique_id, json.dumps({
                    "database": database,
                    "table_name": table_name,
                    "plot_function": plot_function,
                    "form_data": form_data
                }))

                new_url = f"/render-plot/{unique_id}"
                return redirect(new_url)
            else:
                return jsonify({"error": "Invalid plot function selection"}), 400
        else:
            return jsonify({"error": "Plot function not selected"}), 400

    else:  # GET request handling
        print("GET:::::::::::::::::::::::::::::::::")
        return render_template('choose_plot_function_form.html', database=database, table_name=table_name)

'''
@app.route('/view-plot/<database>/<table_name>/<plot_function>', methods=['GET', 'POST'])
def view_plot(database, table_name, plot_function):

    if request.method == "POST":
        print("POST:::::::::::::::::::::::::::::::::")
        # Check if the user has made a plot function choice
        plot_function_choice = request.form.get('plot_choice')
        if plot_function_choice:
            plot_function = plot_function_choice
            if plot_function == "generate_plot_VCR":
                return render_template('input_form_VCR.html', database=database, table_name=table_name, plot_function=plot_function)
            elif plot_function == "generate_plot_TCR_separate":
                return render_template('input_form_2.html', database=database, table_name=table_name, plot_function=plot_function)
            elif plot_function == "generate_plot" or "generate_plot_combined" or "generate_plot_separate":
                return render_template(f'input_form_generate_plot.html', database=database, table_name=table_name, plot_function=plot_function)
            elif plot_function == "generate_plot_64x64":
                return render_template(f'input_form_date.html', database=database, table_name=table_name, plot_function=plot_function)
            elif plot_function == "generate_plot_endurance":
                return render_template(f'input_form_endurance.html', database=database, table_name=table_name, plot_function=plot_function)                
            else:
                return jsonify({"error": "choice not selected"}), 400

        # Ensure plot_function has a value before proceeding
        if plot_function:
            print(f"plot_function: {plot_function}")  # Now plot_function should have a value
            
            # Conditional logic to handle form data based on plot_function
            if plot_function in ["generate_plot", "generate_plot_combined", "generate_plot_separate", "generate_plot_64x64", "generate_plot_VCR", "generate_plot_TCR_separate", "generate_plot_endurance"]:
                form_data_handlers = {
                    "generate_plot": get_form_data_generate_plot,
                    "generate_plot_combined": get_form_data_generate_plot,
                    "generate_plot_separate": get_form_data_generate_plot,
                    "generate_plot_64x64": get_form_data_generate_plot_64x64,
                    "generate_plot_VCR": get_form_data_generate_plot_VCR,
                    "generate_plot_TCR_separate": get_form_data_generate_plot_TCR_separate,
                    "generate_plot_endurance": get_form_data_endurance,
                }
                form_data = form_data_handlers[plot_function](request.form)

                # Store the form data with a unique identifier in Redis
                unique_id = str(uuid.uuid4())
                redis_client.set(unique_id, json.dumps({
                    "database": database,
                    "table_name": table_name,
                    "plot_function": plot_function,
                    "form_data": form_data
                }))

                new_url = f"/render-plot/{unique_id}"
                return redirect(new_url)
            else:
                return jsonify({"error": "Invalid plot function selection"}), 400
        else:
            return jsonify({"error": "Plot function not selected"}), 400

    else:  # GET request handling
        print("GET:::::::::::::::::::::::::::::::::")
        return render_template('choose_plot_function_form.html', database=database, table_name=table_name)
'''
'''
@app.route('/view-plot/<database>/<table_name>/<plot_function>', methods=['GET', 'POST'])
def view_plot(database, table_name, plot_function):

    if request.method == "GET":
        print("GET:::::::::::::::::::::::::::::::::")
        return render_template(f'input_form_generate_plot.html', database=database, table_name=table_name, plot_function=plot_function)
    else:
        #choose the plot_function
        #plot_function = "generate_plot"
        plot_function = "generate_plot_endurance"

        form_data_handlers = {
            "generate_plot": get_form_data_generate_plot,
            "generate_plot_endurance": get_form_data_generate_plot,
        }
        form_data = form_data_handlers[plot_function](request.form)

        # Store the form data with a unique identifier in Redis
        unique_id = str(uuid.uuid4())
        redis_client.set(unique_id, json.dumps({
            "database": database,
            "table_name": table_name,
            "plot_function": plot_function,
            "form_data": form_data
        }))

        new_url = f"/render-plot/{unique_id}"
        return redirect(new_url)
'''
@app.route('/upload-file', methods=['POST'])
def upload_file():
    if 'db_name' not in request.form:
        return jsonify(error="No database selected"), 400

    db_name = request.form['db_name']
    engine = create_db_engine(db_name)

    files = [f for f in request.files.getlist('files[]') if f.filename]
    if not files:
        return jsonify(error="No files selected"), 400

    results = []
    for file in files:
        filename = sanitize_table_name(file.filename)
        file_extension = filename.rpartition('_')[-1]
        print("file_extension:", file_extension)
        file_stream = BytesIO(file.read())

        try:
            df = process_file(file_stream, file_extension, db_name)
            if df.shape[1] > 1017:
                df = df.transpose()
            if not df.empty:
                df.to_sql(filename, engine, if_exists='replace', index=False)
                results.append(f"{filename} uploaded successfully")
            else:
                results.append(f"No data to upload for {filename}. Dataframe is empty.")
        except Exception as e:
            error_msg = f"Error processing {filename}: {str(e)}"
            results.append(error_msg)

    return jsonify(results=results)

@app.route('/delete-record/<database>/<table_name>', methods=['DELETE'])  # delete a table
def delete_record(database, table_name):
    try:
        connection = create_connection(database)
        cursor = connection.cursor()

        query = f"DROP TABLE `{table_name}`"
        cursor.execute(query)

        connection.commit()
        close_connection()

        return "Record deleted successfully", 200
    except mysql.connector.Error as err:
        return str(err), 400

@app.route('/delete-records/<database>', methods=['DELETE'])  # delete multiple tables
def delete_records(database):
    try:
        tables = request.json['tables']
        connection = create_connection(database)
        cursor = connection.cursor()

        for table_name in tables:
            query = f"DROP TABLE `{table_name}`"
            cursor.execute(query)

        connection.commit()
        close_connection()

        return "Records deleted successfully", 200
    except mysql.connector.Error as err:
        return str(err), 400

@app.route('/create-database', methods=['POST'])
def create_database():
    user_name = request.form.get('userName')
    db_name = request.form.get('newDatabaseName')

    # Concatenate user name with database name
    full_db_name = f"{user_name}_{db_name}" if user_name else db_name

    if not full_db_name:
        return jsonify({"error": "No database name provided"}), 400
    success = create_db(full_db_name)  # Use the modified database name
    if success:
        return jsonify({"message": f"Database {full_db_name} created successfully"}), 200
    else:
        return jsonify({"error": "Failed to create database"}), 500

from PIL import Image
@app.route('/download_pptx', methods=['POST'])
def download_pptx():
    '''    
    #template_path = '/home/server/Desktop/device_testing_webapp2/pptx_template/template.pptx'  # Path to the template file
    #template_path = '/home/ubuntu/webapp_2/pptx_template/template.pptx'  #Tetramem EC2 direcotry
    template_path = '/home/server/Desktop/webapp_2/pptx_template/template.pptx'

    plots = request.json.get('plots', [])  # Retrieve the Base64 encoded images from the POST request

    prs = Presentation(template_path)  # Open the template PowerPoint file as the base for the new presentation
    '''
    #aws s3 cp /home/server/Desktop/template.pptx s3://webapp20240318/template.pptx --region us-east-2

    # AWS S3 bucket name and key/path where the PowerPoint template is located
    bucket_name = 'webapp20240318'
    s3_key = 'template.pptx'
    
    # Initialize a boto3 S3 client securely
    s3 = boto3.client('s3', aws_access_key_id='AKIAU62NNZW4ZJJAYZMX', aws_secret_access_key='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')
    
    # Create a BytesIO object to hold the template downloaded from S3
    pptx_template_io = BytesIO()
    
    # Download the template from S3 directly into the BytesIO object
    s3.download_fileobj(bucket_name, s3_key, pptx_template_io)
    
    # Go to the start of the BytesIO object to ensure it can be read correctly
    pptx_template_io.seek(0)
    
    # Load the PowerPoint template
    prs = Presentation(pptx_template_io)
    
    # Retrieve the Base64 encoded images from the POST request
    plots = request.json.get('plots', [])
    
    for plot_data in plots:
        # Decode each Base64 image
        image_data = base64.b64decode(plot_data.split(",")[-1])
        # Open the image for analysis
        image = Image.open(BytesIO(image_data))
        
        # Choose a slide layout (6 is usually a blank slide)
        slide_layout = prs.slide_layouts[6]
        slide = prs.slides.add_slide(slide_layout)
        
        # Remove all shapes (including text boxes) from the slide
        for shape in slide.shapes:
            sp = shape._element
            sp.getparent().remove(sp)
        
        # Get the image size
        img_width, img_height = image.size
        # Get the slide dimensions
        slide_width = prs.slide_width
        slide_height = prs.slide_height
        
        # Calculate the scaling factor to maintain aspect ratio
        ratio = min(slide_width / img_width, slide_height / img_height)
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)
        
        # Center the image
        left = int((slide_width - new_width) / 2)
        top = int((slide_height - new_height) / 2)
        
        # Convert the image data back to a BytesIO object
        img_io = BytesIO(image_data)
        # Add the image to the slide
        slide.shapes.add_picture(img_io, left, top, width=new_width, height=new_height)
    
    # Prepare the presentation to be sent in the response
    pptx_io = BytesIO()
    prs.save(pptx_io)
    pptx_io.seek(0)
    
    # Set up the response with the correct headers
    response = make_response(pptx_io.getvalue())
    response.headers.set('Content-Type', 'application/vnd.openxmlformats-officedocument.presentationml.presentation')
    response.headers.set('Content-Disposition', 'attachment; filename="Downloaded_Presentation.pptx"')
    
    return response

@app.route('/rename-database/<old_name>/<new_name>', methods=['PUT'])
def rename_database_route(old_name, new_name):
    try:
        # Assuming you have a function rename_database in db_operations.py
        # which handles the complex task of renaming a database.
        success = rename_database(old_name, new_name)
        if success:
            return "Database renamed successfully!", 200
        else:
            return "Failed to rename database.", 400
    except mysql.connector.Error as err:
        print("Error occurred:", err)
        return str(err), 500

@app.route('/rename-table/<database>/<old_name>/<new_name>', methods=['PUT'])
def rename_table(database, old_name, new_name):
    try:
        connection = create_connection(database)
        cursor = connection.cursor()

        # Construct and execute the SQL command
        sql = f"RENAME TABLE `{old_name}` TO `{new_name}`"
        cursor.execute(sql)
        connection.commit()
        close_connection()
        return "Table renamed successfully!", 200
    except mysql.connector.Error as err:
        print("Error occurred:", err)
        return str(err), 400

@app.route('/move-tables/<source_db>/<target_db>', methods=['PUT'])
def move_tables_route(source_db, target_db):
    try:
        success = move_tables(source_db, target_db)
        if success:
            return f"Tables moved from {source_db} to {target_db} successfully!", 200
        else:
            return f"Failed to move tables from {source_db} to {target_db}.", 400
    except Exception as err:
        print("Error occurred:", err)
        return str(err), 500

@app.route('/copy-all-tables', methods=['POST'])
def copy_all_tables_route():
    source_db = request.form.get('sourceDatabase')
    target_db = request.form.get('targetDatabase')

    try:
        copy_all_tables(source_db, target_db)
        return jsonify({"message": f"All tables copied from {source_db} to {target_db} successfully"}), 200
    except Exception as err:
        return jsonify({"error": str(err)}), 500

@app.route('/moveSelectedTables', methods=['POST'])
def move_selected_tables():
    data = request.json
    print("Received data for moving tables:", data)
    source_db = data['sourceDb']
    target_db = data['targetDb']
    table_names = data['tableNames']
    response = {"success": [], "failed": []}

    for table_name in table_names:
        try:
            if move_tables_2(source_db, target_db, table_name):
                response["success"].append(table_name)
            else:
                response["failed"].append(table_name)
        except Exception as e:
            print(f"Failed to move {table_name}: {e}")
            response["failed"].append(table_name)

    if response["failed"]:
        return jsonify({"success": False, "message": "Some tables failed to move.", "details": response}), 400
    return jsonify({"success": True, "message": "All selected tables moved successfully."})

@app.route('/copySelectedTables', methods=['POST'])
def copy_selected_tables():
    data = request.json
    print("Received data for copying tables:", data)
    source_db = data['sourceDb']
    target_db = data['targetDb']
    table_names = data['tableNames']
    response = {"success": [], "failed": []}
    print("Attempting to copy tables:", table_names)

    for table_name in table_names:
        try:
            if copy_tables_2(source_db, target_db, table_name):
                response["success"].append(table_name)
            else:
                response["failed"].append(table_name)
        except Exception as e:
            print(f"Failed to copy {table_name}: {e}")
            response["failed"].append(table_name)

    if response["failed"]:
        return jsonify({"success": False, "message": "Some tables failed to copy.", "details": response}), 400
    return jsonify({"success": True, "message": "All selected tables copied successfully."})

@app.route('/getDatabases', methods=['GET'])
def get_databases():
    # Create a database connection
    try:
        conn = create_connection()
        cursor = conn.cursor()
        
        # Use your function to get database names
        databases = get_all_databases(cursor)
        
        # Don't forget to close the cursor and connection when done
        cursor.close()
        conn.close()
        
        # Return the list of databases as a JSON response
        return jsonify(databases)

    except mysql.connector.Error as err:
        # In case of any database connection errors, return an error message
        return jsonify({"error": str(err)}), 500