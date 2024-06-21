# my_flask_app.py
from flask import Flask, request, jsonify, render_template, redirect
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail, Message
from flask_apscheduler import APScheduler
from flask_caching import Cache
from flask_talisman import Talisman  # Ensure Talisman is used if security headers are required

import redis  # Used for Redis client initialization
import json  # Used for JSON handling, if necessary
import uuid  # Used for generating unique identifiers
import random  # Used if random operations are needed

from config import DB_CONFIG, APP_SECRET_KEY  # Import configuration constants

# Initialize Flask application
app = Flask(__name__)

# Configure mail service
app.config.update(
    MAIL_SERVER='smtp.gmail.com',
    MAIL_PORT=587,
    MAIL_USE_TLS=True,
    MAIL_USERNAME='victor.hsiao@tetramem.com',  # sender appear to others
    MAIL_PASSWORD='nmcgzlmzyuwdpjbs',  # Your generated app-specific password without spaces
    MAIL_DEFAULT_SENDER='Automatic notification<victor.hsiao@tetramem.com>'  # Optional: Include a name
)
mail = Mail(app)

# Initialize Redis client
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# Configure caching
app.config['CACHE_TYPE'] = 'SimpleCache'  # Using simple in-memory cache for demonstration
app.config['CACHE_DEFAULT_TIMEOUT'] = 300  # Cache timeout in seconds
cache = Cache(app)

# Configure server-side session storage
app.config['SECRET_KEY'] = APP_SECRET_KEY
app.config['SQLALCHEMY_DATABASE_URI'] = (
    f"mysql+pymysql://{DB_CONFIG['DB_USER']}:{DB_CONFIG['MYSQL_PASSWORD_RAW']}@{DB_CONFIG['DB_HOST']}/users"
)

# Initialize APScheduler
scheduler = APScheduler()
scheduler.init_app(app)
scheduler.start()
