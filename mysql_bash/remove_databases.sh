#!/bin/bash

#RDS_ENDPOINT="webapp.cdecmst6qwog.us-east-2.rds.amazonaws.com"
#RDS_USERNAME="admin"
#RDS_PASSWORD="Aa11720151015"

RDS_ENDPOINT="localhost"
RDS_USERNAME="root"
RDS_PASSWORD="Str0ng_P@ssw0rd!"

# Exclude system databases
EXCLUDE_DB="^(Database|information_schema|performance_schema|mysql|sys)$"

# Get the list of databases
databases=`mysql -h $RDS_ENDPOINT -u $RDS_USERNAME -p$RDS_PASSWORD -e "SHOW DATABASES;" | grep -Ev "$EXCLUDE_DB"`

echo "Removing databases..."

# Loop through and drop each database
for db in $databases; do
    echo "Dropping database $db"
    mysql -h $RDS_ENDPOINT -u $RDS_USERNAME -p$RDS_PASSWORD -e "DROP DATABASE $db"
done

echo "All databases have been removed."

#run thie scrip with: bash remove_databases.sh

