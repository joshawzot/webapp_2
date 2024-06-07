#!/bin/bash

# Local EC2 MySQL credentials
HOST='localhost'
USER='root'
PASSWORD='password'

# Temporary file to store the database list
DB_LIST='db_list.txt'

# Command to list all databases and exclude system databases
mysql -h "$HOST" -u "$USER" -p"$PASSWORD" -e "SHOW DATABASES;" | grep -Ev "(Database|information_schema|performance_schema|mysql|sys)" > "$DB_LIST"

# Loop through each database and dump it
while read DB; do
    #mysqldump -h "$HOST" -P 3306 -u "$USER" -p"$PASSWORD" --databases "$DB" >> all_databases_backup.sql
    #Add date at the end of the dump file and specify dump directory
    mysqldump -h "$HOST" -P 3306 -u "$USER" -p"$PASSWORD" --databases "$DB" >> "/home/ubuntu/backup_sql/all_databases_backup_$(date +\%Y-\%m-\%d).sql"

done < "$DB_LIST"

# Clean up
rm "$DB_LIST"

#make a bash executable: chmod +x scriptname.sh
#bash dump_databases_avoid_system_files.sh

#chmod +x dump_databases_avoid_system_files.sh && ./dump_databases_avoid_system_files.sh