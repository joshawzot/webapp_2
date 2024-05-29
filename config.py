# config.py
from urllib.parse import quote_plus

# Secret Keys
APP_SECRET_KEY = '1234'

#eng
#ENG = False
ENG = True

#Analyze type
#MULTI_DATABASE_ANALYSIS = True
MULTI_DATABASE_ANALYSIS = False

# Switch for local or RDS MySQL
#LOCAL_DB = False  # Set to False for using AWS RDS
LOCAL_DB = True

# Initialize DB_CONFIG
DB_CONFIG = {}

# Local mysql on EC2 Database Configuration
if LOCAL_DB:
    DB_CONFIG['RDS_PORT'] = None
    DB_CONFIG['DB_HOST'] = "localhost"
    DB_CONFIG['DB_USER'] = "root"
    DB_CONFIG['MYSQL_PASSWORD_RAW'] = 'password'

# Default Remote Database Configuration
else:
    DB_CONFIG['RDS_PORT'] = 3306
    DB_CONFIG['DB_HOST'] = 'webapp.cdecmst6qwog.us-east-2.rds.amazonaws.com'
    DB_CONFIG['DB_USER'] = 'admin'
    DB_CONFIG['MYSQL_PASSWORD_RAW'] = 'Aa11720151015'

# Local mysql on lenovoi7
'''
if LOCAL_DB:
    DB_CONFIG['RDS_PORT'] = None  # Implicitly defaults to 3306
    DB_CONFIG['DB_HOST'] = "localhost"
    DB_CONFIG['DB_USER'] = "root"
    DB_CONFIG['MYSQL_PASSWORD_RAW'] = '' 
'''

#from local
#mysql -h localhost -u root -p or mysql -u root -ppassword

#add inbound rules 3306 to ec2

#from anywhere
#mysql -h webapp.cdecmst6qwog.us-east-2.rds.amazonaws.com -u admin -pAa11720151015 -P 3306
#telnet webapp.cdecmst6qwog.us-east-2.rds.amazonaws.com 3306
#telnet 18.189.54.37 3306   #18.189.54.37 is public ipv4 address

#mysql -h <mysql-server-ip> -u username -p      #mysql -h 18.189.54.37 -u remote_user -pAa@11720151015
#mysql -u root -ppassword
#CREATE USER 'remote_user'@'%' IDENTIFIED BY 'Aa@11720151015';
#GRANT ALL PRIVILEGES ON *.* TO 'remote_user'@'%' WITH GRANT OPTION;
#FLUSH PRIVILEGES;


# Common for all configurations
DB_CONFIG['MYSQL_PASSWORD'] = quote_plus(DB_CONFIG['MYSQL_PASSWORD_RAW'])
