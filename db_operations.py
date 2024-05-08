# db_operations.py
import mysql.connector
from config import DB_CONFIG  # must have

connection = None

''''def create_connection(database=None):
    global connection
    if connection is None or not connection.is_connected():
        connection = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=MYSQL_PASSWORD,
            database=database
        )
    return connection'''

from mysql.connector import pooling
# Initialize the connection pool using DB_CONFIG values
connection_pool = mysql.connector.pooling.MySQLConnectionPool(
    pool_name="mypool",
    pool_size=10,  # Adjust this value based on your requirements
    host=DB_CONFIG['DB_HOST'],
    user=DB_CONFIG['DB_USER'],
    password=DB_CONFIG['MYSQL_PASSWORD_RAW']  # Using encrypted password
)

def create_connection(database=None):
    global connection_pool
    try:
        connection = connection_pool.get_connection()
    except mysql.connector.errors.PoolError:
        # The pool is exhausted; create a new connection
        connection = mysql.connector.connect(
            host=DB_CONFIG['DB_HOST'],
            user=DB_CONFIG['DB_USER'],
            password=DB_CONFIG['MYSQL_PASSWORD_RAW'],  # Using encrypted password
            database=database
        )
    else:
        if database is not None:
            cursor = connection.cursor()
            cursor.execute(f"USE {database};")
            cursor.close()
    return connection

def create_db(db_name):
    # Get a connection from the connection pool
    connection = create_connection()
    cursor = connection.cursor()
    try:
        # Create database with the specified name
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name};")
        print(f"Database {db_name} created successfully.")
        return True  # Indicate success
    except mysql.connector.Error as err:
        # Handle errors in database creation
        print(f"Failed creating database: {err}")
        return False  # Indicate failure
    finally:
        # Close cursor and release connection back to the pool
        cursor.close()
        close_connection()

def fetch_data(cursor, query):
    cursor.execute(query)
    return cursor.fetchall()

def close_connection():
    global connection
    if connection is not None and connection.is_connected():
        connection.close()
        connection = None

from sqlalchemy import create_engine
''''def create_db_engine():
    engine = create_engine(f"mysql+mysqlconnector://{DB_USER}:{MYSQL_PASSWORD}@{DB_HOST}/{db_name}")
    return engine'''

def create_db_engine(db_name):
    engine_url = f"mysql+mysqlconnector://{DB_CONFIG['DB_USER']}:{DB_CONFIG['MYSQL_PASSWORD']}@{DB_CONFIG['DB_HOST']}/{db_name}"
    engine = create_engine(
        engine_url,
        pool_size=10,  # Maximum number of connections to keep in the pool
        max_overflow=5  # Allow up to 5 additional connections beyond pool_size
    )
    return engine

def get_all_databases(cursor):
    """Fetch all database names from the MySQL server and return them as a list, excluding restricted databases."""
    # Define your restricted patterns or names
    restricted_patterns = ['performance_schema', 'mysql', 'information_schema', 'sys']
    
    try:
        cursor.execute("SHOW DATABASES")
        # Use list comprehension to extract database names from the cursor
        all_databases = [db[0] for db in cursor]

        # Filter your databases list to exclude restricted databases
        filtered_databases = [db for db in all_databases if db not in restricted_patterns]

        return filtered_databases

    except mysql.connector.Error as err:
        print(f"Failed to list databases: {err}")
        # Decide how to handle the error. Here we're returning an empty list, but you might want to re-raise the error or handle it differently.
        return []

def connect_to_db(user, password, host, port=None):
    """Connect to the MySQL server and return the connection."""
    connection_params = {
        'host': host,
        'user': user,
        'password': password
    }
    if port:
        connection_params['port'] = port
    try:
        return mysql.connector.connect(**connection_params)
    except mysql.connector.Error as e:
        print(f"Error connecting to the database: {e}")
        return None  # Return None if there's a connection error

def fetch_tables(database):
    """Fetch table names, creation times, and dimensions from the database."""
    connection = create_connection(database)
    cursor = connection.cursor()

    # Query to fetch table names and creation times
    table_query = """
    SELECT TABLE_NAME, CREATE_TIME
    FROM information_schema.tables
    WHERE table_schema = %s
    ORDER BY CREATE_TIME DESC;
    """
    cursor.execute(table_query, (database,))
    table_info = cursor.fetchall()

    # Dictionary to store table information with dimensions
    tables = []
    for name, time in table_info:
        # Count the number of columns
        column_query = """
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s;
        """
        cursor.execute(column_query, (database, name))
        column_count = cursor.fetchone()[0]

        # Count the number of rows
        row_query = f"SELECT COUNT(*) FROM `{name}`;"
        cursor.execute(row_query)
        row_count = cursor.fetchone()[0]

        # Store table info
        tables.append({'table_name': name, 'creation_time': time, 'dimensions': f"{row_count}x{column_count}"})

    cursor.close()
    connection.close()
    return tables

def rename_database(old_name, new_name):
    connection = create_connection()
    cursor = connection.cursor()

    try:
        # Create new database
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{new_name}`;")

        # Fetch all tables from the old database
        cursor.execute(f"SHOW TABLES FROM `{old_name}`;")
        tables = cursor.fetchall()

        # Move each table to the new database
        for (table_name,) in tables:
            cursor.execute(f"RENAME TABLE `{old_name}`.`{table_name}` TO `{new_name}`.`{table_name}`;")

        # Drop old database
        cursor.execute(f"DROP DATABASE `{old_name}`;")

        # Commit the changes
        connection.commit()
        return True

    except mysql.connector.Error as err:
        print(f"Error while renaming database: {err}")
        connection.rollback()  # Rollback in case of any error
        return False
    finally:
        cursor.close()
        connection.close()

'''def move_tables(source_db, target_db):
    connection = create_connection()
    cursor = connection.cursor()

    try:
        # Fetch all tables from the source database
        cursor.execute(f"SHOW TABLES FROM `{source_db}`;")
        tables = cursor.fetchall()

        # Move each table to the target database
        for (table_name,) in tables:
            cursor.execute(f"RENAME TABLE `{source_db}`.`{table_name}` TO `{target_db}`.`{table_name}`;")

        # Commit the changes
        connection.commit()
        return True

    except mysql.connector.Error as err:
        print(f"Error while moving tables: {err}")
        connection.rollback()  # Rollback in case of any error
        return False
    finally:
        cursor.close()
        connection.close()'''

def copy_all_tables(source_db, target_db):
    # Connect to source database
    source_connection = create_connection(source_db)
    source_cursor = source_connection.cursor()

    # Connect to target database
    target_connection = create_connection(target_db)
    target_cursor = target_connection.cursor()

    try:
        # Get all table names from the source database
        source_cursor.execute(f"SHOW TABLES")
        tables = source_cursor.fetchall()

        for (table_name,) in tables:
            # Create each table in the target database
            source_cursor.execute(f"SHOW CREATE TABLE `{table_name}`")
            create_table_query = source_cursor.fetchone()[1]
            create_table_query = create_table_query.replace(f"`{table_name}`", f"`{target_db}`.`{table_name}`")
            target_cursor.execute(create_table_query)

            # Copy the data
            source_cursor.execute(f"SELECT * FROM `{table_name}`")
            data = source_cursor.fetchall()
            columns = ", ".join([f"`{desc[0]}`" for desc in source_cursor.description])
            insert_query = f"INSERT INTO `{target_db}`.`{table_name}` ({columns}) VALUES (%s" + ", %s" * (len(source_cursor.description) - 1) + ")"
            target_cursor.executemany(insert_query, data)

        # Commit the changes
        target_connection.commit()
    except mysql.connector.Error as err:
        print(f"Error while copying tables: {err}")
        target_connection.rollback()
        raise
    finally:
        source_cursor.close()
        target_cursor.close()
        source_connection.close()
        target_connection.close()

def move_tables(source_db, target_db, table_names):
    connection = create_connection()
    cursor = connection.cursor()

    try:
        # Move each specified table to the target database
        for table_name in table_names:
            print("table_name:", table_name)
            cursor.execute(f"RENAME TABLE `{source_db}`.`{table_name}` TO `{target_db}`.`{table_name}`;")

        # Commit the changes
        connection.commit()
        return True

    except mysql.connector.Error as err:
        print(f"Error while moving tables: {err}")
        connection.rollback()  # Rollback in case of any error
        return False
    finally:
        cursor.close()
        connection.close()

def copy_tables(source_db, target_db, table_names):
    source_connection = create_connection(database=source_db)
    target_connection = create_connection(database=target_db)
    source_cursor = source_connection.cursor()
    target_cursor = target_connection.cursor()

    try:
        for table_name in table_names:
            # Fetch the CREATE TABLE statement for each table
            source_cursor.execute(f"SHOW CREATE TABLE `{table_name}`;")
            create_table_query = source_cursor.fetchone()[1]

            # Replace the source database name with the target database name in the CREATE TABLE statement
            create_table_query_modified = create_table_query.replace(f"`{source_db}`.", f"`{target_db}`.")

            # Create the table in the target database
            target_cursor.execute(create_table_query_modified)

            # Copy data from each table in the source database to the target database
            source_cursor.execute(f"SELECT * FROM `{table_name}`;")
            rows = source_cursor.fetchall()
            columns_count = len(source_cursor.description)
            placeholders = ', '.join(['%s'] * columns_count)
            target_cursor.executemany(f"INSERT INTO `{target_db}`.`{table_name}` VALUES ({placeholders})", rows)

        # Commit the changes to the target database
        target_connection.commit()
        return True

    except mysql.connector.Error as err:
        print(f"Error while copying tables: {err}")
        # Rollback the target connection in case of an error
        target_connection.rollback()
        return False
    finally:
        source_cursor.close()
        source_connection.close()
        target_cursor.close()
        target_connection.close()

# Assume you have a function to get a connection from a pool or create a new one if the pool is empty
def get_db_connection(database=None):
    try:
        connection = mysql.connector.connect(
            pool_name="mypool",
            host=DB_CONFIG['DB_HOST'],
            user=DB_CONFIG['DB_USER'],
            password=DB_CONFIG['MYSQL_PASSWORD'],  # Assuming password is already appropriately handled
            database=database
        )
        if database:
            cursor = connection.cursor()
            cursor.execute(f"USE {database};")
            cursor.close()
        return connection
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

def copy_tables_2(source_db, target_db, table_name):
    """Copies a table from one database to another on the same MySQL server."""
    conn = get_db_connection()
    if conn is None:
        return False

    cursor = conn.cursor()
    try:
        # Verify if the table exists in the source database
        cursor.execute(f"SHOW TABLES FROM `{source_db}` LIKE '{table_name}';")
        if not cursor.fetchone():
            print(f"Table {table_name} does not exist in database {source_db}.")
            return False

        # Ensure the target database exists
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{target_db}`;")
        conn.commit()

        # Copy table structure from source to target database
        cursor.execute(f"CREATE TABLE `{target_db}`.`{table_name}` LIKE `{source_db}`.`{table_name}`;")
        
        # Copy data from source to target database
        cursor.execute(f"INSERT INTO `{target_db}`.`{table_name}` SELECT * FROM `{source_db}`.`{table_name}`;")
        conn.commit()
        print(f"Table {table_name} copied successfully from {source_db} to {target_db}.")
        return True
    except Error as e:
        print(f"Failed to copy table {table_name} from {source_db} to {target_db}: {e}")
        conn.rollback()  # Ensure to rollback on error
        return False
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def move_tables_2(source_db, target_db, table_name):
    """Moves a table from one database to another on the same MySQL server."""
    conn = get_db_connection()
    if conn is None:
        return False

    cursor = conn.cursor()
    try:
        # Check if the target database exists, create if not
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{target_db}`;")
        conn.commit()

        # Check if the table exists in the source database
        cursor.execute(f"SHOW TABLES FROM `{source_db}` LIKE '{table_name}';")
        if not cursor.fetchone():
            print(f"Table {table_name} does not exist in database {source_db}.")
            return False

        # Move table from source to target database
        cursor.execute(f"RENAME TABLE `{source_db}`.`{table_name}` TO `{target_db}`.`{table_name}`;")
        conn.commit()
        print(f"Table {table_name} moved successfully from {source_db} to {target_db}.")
        return True
    except Error as e:
        print(f"Failed to move table {table_name} from {source_db} to {target_db}: {e}")
        conn.rollback()  # Rollback in case of errors
        return False
    finally:
        cursor.close()
        conn.close()

import csv
import io

def get_csv_from_table(database, table_name):
    connection = create_connection(database)
    if connection is None:
        raise Exception("Failed to connect to the database.")

    cursor = connection.cursor()
    try:
        # Construct the SQL query to fetch all data from the specified table
        query = f"SELECT * FROM `{table_name}`;"
        cursor.execute(query)

        # Use StringIO to capture CSV output
        output = io.StringIO()
        csv_writer = csv.writer(output)

        # Write header (column names)
        column_headers = [i[0] for i in cursor.description]
        csv_writer.writerow(column_headers)

        # Write data rows
        for row in cursor.fetchall():
            csv_writer.writerow(row)

        # Get CSV string from StringIO
        csv_string = output.getvalue()
        output.close()

        return csv_string
    except Exception as e:
        print(f"Error fetching data from table {table_name}: {e}")
        return None
    finally:
        cursor.close()
        connection.close()