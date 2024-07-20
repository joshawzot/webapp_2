# my_flask_app.py
from flask_mail import Mail, Message
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

# Initialize APScheduler
scheduler = APScheduler()
scheduler.init_app(app)
scheduler.start()

@app.before_first_request
def initialize_scheduler():
    # Schedule the database schema check to run every 10 minutes
    scheduler.add_job(id='Schema Monitor', func=check_for_new_schemas, trigger='interval', minutes=10)

app.config.update(
    MAIL_SERVER='smtp.example.com',
    MAIL_PORT=587,
    MAIL_USE_TLS=True,
    MAIL_USERNAME='your-email@example.com',
    MAIL_PASSWORD='your-password',
    MAIL_DEFAULT_SENDER='your-email@example.com'
)
mail = Mail(app)

redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

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
app.config['SQLALCHEMY_DATABASE_URI'] = (
    f"mysql+pymysql://{DB_CONFIG['DB_USER']}:{DB_CONFIG['MYSQL_PASSWORD_RAW']}@{DB_CONFIG['DB_HOST']}/users"
)

'''
db = SQLAlchemy(app)  # database interactions, not used yet
'''

# Import routes
import route_handlers

'''@app.route('/')  # Defines a route for the root URL
def home():
    return redirect(url_for('home_page'))'''


