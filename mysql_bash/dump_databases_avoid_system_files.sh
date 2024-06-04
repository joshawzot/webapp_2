#!/bin/bash

# Local EC2 MySQL credentials
HOST='localhost'
USER='root'
PASSWORD='password'

# Temporary file to store the database list
DB_LIST='db_list.txt'

# Backup directory
BACKUP_DIR="/home/ubuntu/backup_sql"

# Ensure the backup directory exists
mkdir -p "$BACKUP_DIR"

# Command to list all databases and exclude system databases
mysql -h "$HOST" -u "$USER" -p"$PASSWORD" -e "SHOW DATABASES;" | grep -Ev "(Database|information_schema|performance_schema|mysql|sys)" > "$DB_LIST"

# Current date and time for filename
CURRENT_DATE=$(date +%Y-%m-%d_%H-%M-%S)

# Loop through each database and dump it
while read DB; do
    mysqldump -h "$HOST" -P 3306 -u "$USER" -p"$PASSWORD" --databases "$DB" > "$BACKUP_DIR/all_databases_backup_${CURRENT_DATE}.sql"
done < "$DB_LIST"

# Remove all backups except the most recent one
find $BACKUP_DIR -type f -name 'all_databases_backup_*.sql' ! -name "all_databases_backup_${CURRENT_DATE}.sql" -delete

# Download file to local machine  #not ready yet since the VPN is not setup for linux user
#scp -i ~/.ssh/id_rsa_lenovoi7 -P 22 "${BACKUP_DIR}/all_databases_backup_${CURRENT_DATE}.sql" ubuntu@192.168.68.164:/home/ubuntu/Downloads/

# Clean up
rm "$DB_LIST"


#make a bash executable: chmod +x scriptname.sh
#bash dump_databases_avoid_system_files.sh

#chmod +x dump_databases_avoid_system_files.sh && ./dump_databases_avoid_system_files.sh