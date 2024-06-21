# run.py

from werkzeug.serving import make_server  # can't use https yet, if want to use https, has to use server like Nginx or Apache
#from my_flask_app import app

from apscheduler.schedulers.background import BackgroundScheduler

# Initialize scheduler
scheduler = BackgroundScheduler()

def run_flask():
    server = make_server('0.0.0.0', 5000, app, threaded=True)  # Enable multi-threading here
    server.serve_forever()

if __name__ == "__main__":  #in app.py, there is "app = Flask(__name__) "
    # This block of code will only execute when the script is run directly.
    from route_handlers import app  # import here to avoid circular dependency
    #scheduler.start()
    run_flask()