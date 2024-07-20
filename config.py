# config.py
from urllib.parse import quote_plus

# Secret Keys
APP_SECRET_KEY = '1234'

#eng
ENG = False
#ENG = True

# Initialize DB_CONFIG
DB_CONFIG = {}

# Local mysql on lenovoi7
DB_CONFIG['RDS_PORT'] = None  # Implicitly defaults to 3306
DB_CONFIG['DB_HOST'] = "localhost"
DB_CONFIG['DB_USER'] = "root"
DB_CONFIG['MYSQL_PASSWORD_RAW'] = ''

#for remote user on other machine
'''DB_CONFIG['RDS_PORT'] = 3306  # Implicitly defaults to 3306
DB_CONFIG['DB_HOST'] = "192.168.68.164"
DB_CONFIG['DB_USER'] = "remote_user"
DB_CONFIG['MYSQL_PASSWORD_RAW'] = '' '''

# Common for all configurations
DB_CONFIG['MYSQL_PASSWORD'] = quote_plus(DB_CONFIG['MYSQL_PASSWORD_RAW'])
