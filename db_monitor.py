import mysql.connector
from flask_mail import Message
from my_flask_app import mail, app  # Assuming Flask-Mail is set up in your main app module
from config import *
import datetime

def check_for_new_schemas():
    print("check_for_new_schemas")
    with app.app_context():
        cnx = mysql.connector.connect(
            host=DB_CONFIG['DB_HOST'],
            user=DB_CONFIG['DB_USER'],
            password=DB_CONFIG['MYSQL_PASSWORD_RAW'],
            database='mysql'
        )
        cursor = cnx.cursor()

        # Calculate the time one minute ago from now
        one_minute_ago = datetime.datetime.now() - datetime.timedelta(minutes=1)
        one_minute_ago = one_minute_ago.strftime('%Y-%m-%d %H:%M:%S')

        query = f"""
        SELECT argument FROM general_log
        WHERE command_type = 'Query'
        AND event_time >= '{one_minute_ago}'
        AND (
            argument LIKE '%CREATE DATABASE%' OR 
            argument LIKE '%CREATE SCHEMA%' OR 
            argument LIKE '%DROP DATABASE%' OR 
            argument LIKE '%DROP SCHEMA%' OR 
            argument LIKE '%ALTER TABLE%' OR 
            argument LIKE '%DROP TABLE%'
        )
        AND argument NOT LIKE '%general_log%'
        """

        cursor.execute(query)
        results = cursor.fetchall()
        if results:
            for (command,) in results:
                if isinstance(command, bytes):
                    command = command.decode('utf-8')
                print(f"Sending email for command: {command}")
                send_email(
                    "Schema Creation Alert",
                    f"A new schema was created with the command: {command}",
                    ['victor.hsiao@tetramem.com', 'max.zhang@tetramem.com', 'mingyi.rao@tetramem.com', 'mingche.wu@tetramem.com']
                )
        else:
            print("No new schemas found.")
        cursor.close()
        cnx.close()

def send_email(subject, message, recipients):
    with app.app_context():
        msg = Message(subject, sender=app.config['MAIL_DEFAULT_SENDER'], recipients=recipients)
        msg.body = message
        mail.send(msg)
