import mysql.connector
from config import DB_CONFIG  # Use the consolidated DB configuration

def create_database(cursor, db_name):
    """Initialize a new MySQL database."""
    try:
        cursor.execute(f"CREATE DATABASE {db_name}")
        print(f"Database {db_name} created successfully.")
    except mysql.connector.Error as err:
        print(f"Failed creating database: {err}")
        exit(1)

def create_database_setup(database_name):
    """Set up the database using configurations from DB_CONFIG."""
    try:
        # Establish connection using the settings from DB_CONFIG
        with mysql.connector.connect(host=DB_CONFIG['DB_HOST'],
                                     user=DB_CONFIG['DB_USER'],
                                     password=DB_CONFIG['MYSQL_PASSWORD'],
                                     port=DB_CONFIG.get('RDS_PORT')) as conn, conn.cursor() as cursor:
            try:
                cursor.execute(f"USE {database_name}")
            except mysql.connector.Error as err:
                print(f"Database {database_name} does not exist.")
                if err.errno == mysql.connector.errorcode.ER_BAD_DB_ERROR:
                    create_database(cursor, database_name)
                    conn.database = database_name
                else:
                    print('Exiting')
                    print(err)
                    exit(1)
    except Exception as e:
        print('Exiting')
        print(f"Error connecting to the database: {e}")
        exit(1)

# Main code
database_names = ['boxplot_histogram_cdf']

for database_name in database_names:
    create_database_setup(database_name)