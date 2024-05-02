# config.py
from urllib.parse import quote_plus

# Secret Keys
APP_SECRET_KEY = '1234'

#eng
#ENG = False
ENG = True

#Analyze type
MULTI_DATABASE_ANALYSIS = True
#MULTI_DATABASE_ANALYSIS = False

# Switch for local or RDS MySQL
LOCAL_DB = False  # Set to False for using AWS RDS
#LOCAL_DB = True

# Initialize DB_CONFIG
DB_CONFIG = {}

# Local Database Configuration
if LOCAL_DB:
    DB_CONFIG['RDS_PORT'] = None
    DB_CONFIG['DB_HOST'] = "localhost"
    DB_CONFIG['DB_USER'] = "root"
    DB_CONFIG['MYSQL_PASSWORD_RAW'] = 'Str0ng_P@ssw0rd!'

# Default Remote Database Configuration
else:
    DB_CONFIG['RDS_PORT'] = 3306
    DB_CONFIG['DB_HOST'] = 'webapp.cdecmst6qwog.us-east-2.rds.amazonaws.com'
    DB_CONFIG['DB_USER'] = 'admin'
    DB_CONFIG['MYSQL_PASSWORD_RAW'] = 'Aa11720151015'

#check the connectivity to AWS RDS
#telnet webapp.cdecmst6qwog.us-east-2.rds.amazonaws.com 3306

# Common for all configurations
DB_CONFIG['MYSQL_PASSWORD'] = quote_plus(DB_CONFIG['MYSQL_PASSWORD_RAW'])
