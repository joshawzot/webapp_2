# my_flask_app.py
from flask import Flask, after_this_request
from flask_talisman import Talisman
from flask_sqlalchemy import SQLAlchemy
from flask_caching import Cache
from config import DB_CONFIG, APP_SECRET_KEY # must have

import redis
import json
from flask import Flask, request, jsonify, render_template, redirect
import uuid
import random

app = Flask(__name__) # This creates an instance of the Flask class named 'app'

redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
'''
The Redis server is the process that manages the Redis in-memory data store. 
It listens for connections from clients on a specified port (by default, 6379) and processes commands sent by those clients.
When you specify host='localhost', it means the Redis server is running on the same machine as your Python application.
'''
'''
so the redis data is sent from the client side to the server side and then stored on the RAM of the server?  Yes
'''

app.config['CACHE_TYPE'] = 'SimpleCache'  # Using simple in-memory cache for demonstration
'''
This line configures the type of cache to use. SimpleCache is a basic in-memory cache provided by Flask-Caching. 
It's suitable for single-process applications and is often used for development and testing due to its simplicity. 
Other cache types supported by Flask-Caching include Memcached, Redis, and FileSystemCache, among others.
'''
app.config['CACHE_DEFAULT_TIMEOUT'] = 300  # Cache timeout in seconds
cache = Cache(app)
'''
This line creates an instance of the Cache class, passing in the Flask application app as an argument. 
This effectively initializes the caching system with the configurations specified in app.config (the cache type and default timeout). 
After this, cache can be used to cache the output of views or other functions within your Flask application.
'''

# Configure server-side session storage
app.config['SECRET_KEY'] = APP_SECRET_KEY

'''csp = {
    'default-src': [
        '\'self\'',
        'maxcdn.bootstrapcdn.com',
        'code.jquery.com',
        'cdnjs.cloudflare.com',
    ],
    'script-src': [
        '\'self\'',
        "'unsafe-inline'",  # Allows inline scripts and event handlers
        'maxcdn.bootstrapcdn.com',
        'code.jquery.com',
        'cdnjs.cloudflare.com',
    ],
    'style-src': [
        '\'self\'',
        'maxcdn.bootstrapcdn.com',
        "'unsafe-inline'",  # Correct way to allow inline styles
    ],
    'img-src': [
        '\'self\'', 
        'data:', 
        'https://d2lsypbr1lit3e.cloudfront.net'
    ],
    'font-src': [
        '\'self\'', 
        'maxcdn.bootstrapcdn.com'
    ],
}

talisman = Talisman(
    app,
    content_security_policy=csp,
    force_https=False
)

#talisman forces http to https
#avoid https but still making talisman working

'''

'''Flask-Talisman is a Flask extension that enhances the security of your Flask application by managing HTTP security headers. It's designed to make it easier to implement various security practices that protect your app from a range of common vulnerabilities and attacks. Here's an overview of what Flask-Talisman does:
1. Content Security Policy (CSP)

One of the primary features of Flask-Talisman is its support for Content Security Policy (CSP), which is a security layer that helps prevent Cross-Site Scripting (XSS) and data injection attacks. CSP allows you to specify which domains the browser should consider as valid sources for executable scripts, stylesheets, images, etc., for your site. By defining a strict CSP, you can significantly reduce the risk of XSS attacks.
2. HTTPS Enforcement

Flask-Talisman can be configured to enforce HTTPS for all requests, redirecting HTTP requests to HTTPS. This ensures that communications between your users and your web application are encrypted, protecting against eavesdroppers and man-in-the-middle attacks. It also sets the Strict-Transport-Security header, which tells browsers to always use HTTPS for your site, further enhancing security.
'''

#app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'

# This line is specifically for connecting to an SQLite database. SQLite is a file-based database system, 
#and the connection string points to a file on your filesystem (users.db). 
#SQLite does not require a server to be running, and the connection is made directly to the database file.

'''
User
Do I have to install SQLite?

No, you do not have to install SQLite separately if you are using it within a Python environment, 
especially when working with frameworks like Flask. 
SQLite comes bundled with Python, starting from Python 2.5 and onwards. 
This means that the SQLite database engine and the Python module for interacting with SQLite (sqlite3) are included with your Python installation, 
making it immediately available for use without any additional installation steps.

SQLite is a lightweight, file-based database system that is designed to be self-contained, 
serverless, and zero-configuration, which makes it an attractive choice for development, 
testing, and even production use for smaller projects or applications with simple database needs.

However, if you're working on a project that requires a more robust database system with features such as concurrency control, 
advanced security, or the ability to handle large volumes of transactions and data, 
you might opt for a more powerful database system like MySQL or PostgreSQL. In those cases, 
you would need to install the database system separately and ensure it's running either on your development machine or a remote server. 
Additionally, you would need to install the appropriate Python library (e.g., PyMySQL for MySQL, psycopg2 for PostgreSQL) 
to enable your Flask application to communicate with the database.

To summarize, for SQLite:

    No Installation Required: SQLite and its Python interface come pre-installed with Python.
    Immediate Use: You can start using SQLite in your Flask applications right away by importing the sqlite3 module and defining your database URI as shown in your SQLite configuration example.

For other database systems like MySQL:

    Installation Required: You need to install the database server and the Python client library for the database system you choose to use.
    Configuration Needed: You'll have to configure your Flask application to connect to the database server using the appropriate connection URI and client library.
'''

app.config['SQLALCHEMY_DATABASE_URI'] = (
    f"mysql+pymysql://{DB_CONFIG['DB_USER']}:{DB_CONFIG['MYSQL_PASSWORD_RAW']}@{DB_CONFIG['DB_HOST']}/users"
)

'''
This line is for connecting to a MySQL database using the pymysql driver. 
MySQL is a server-based relational database management system (RDBMS). 
The connection string includes the username, password, 
host (which could be an IP address or a domain name), 
and the specific database name (users) to which the connection is made. 
This setup requires a MySQL server to be running and accessible at the specified host.
SQLite connections are made to a file directly, while MySQL connections are made to a database server over a network protocol.
'''

'''
MySQL operates in a client-server model. 
The MySQL server can be installed on the same machine as your application (locally) or on a different machine (remotely) that can be accessed over a network.
Because of its client-server architecture, a MySQL database can be hosted on a remote server, making it accessible to applications running on different machines. 
'''

'''
from flask_bcrypt import Bcrypt
bcrypt = Bcrypt(app)  # password hashing, not used yet

db = SQLAlchemy(app)  # database interactions, not used yet
'''

# Import routes
import route_handlers

'''@app.route('/')  # Defines a route for the root URL
def home():
    return redirect(url_for('home_page'))'''


