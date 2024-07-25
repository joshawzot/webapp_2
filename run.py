'''# run.py

from werkzeug.serving import make_server  # can't use https yet, if want to use https, has to use server like Nginx or Apache
#from my_flask_app import app

from config import*

def run_flask():
    server = make_server('0.0.0.0', 5000, app, threaded=True)  # Enable multi-threading here
    server.serve_forever()

if __name__ == "__main__":  #in app.py, there is "app = Flask(__name__) "
    # This block of code will only execute when the script is run directly.
    from route_handlers import app  # import here to avoid circular dependency
    run_flask()'''
'''
# run.py
import threading
from werkzeug.serving import make_server  # can't use https yet, if want to use https, has to use server like Nginx or Apache
#from my_flask_app import app

from config import *

def run_flask(port):
    server = make_server('0.0.0.0', port, app, threaded=True)  # Enable multi-threading here
    server.serve_forever()

if __name__ == "__main__":  #in app.py, there is "app = Flask(__name__) "
    from route_handlers import app  # import here to avoid circular dependency
    
    ports = [5000, 4000, 3000, 2000]
    threads = []

    for port in ports:
        thread = threading.Thread(target=run_flask, args=(port,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()
'''

#To ensure complete isolation between different ports, you can use multiprocessing instead of threading. Each process will have its own memory space, which will help avoid interference.
# run.py
from multiprocessing import Process
from werkzeug.serving import make_server  # can't use https yet, if want to use https, has to use server like Nginx or Apache
#from my_flask_app import app

from config import *

def run_flask(port):
    from route_handlers import app  # Import within the function to avoid circular dependency and shared state issues
    server = make_server('0.0.0.0', port, app, threaded=True)  # Enable multi-threading here
    server.serve_forever()

if __name__ == "__main__":  # in app.py, there is "app = Flask(__name__) "
    ports = [5000, 4000, 3000, 2000]
    processes = []

    for port in ports:
        process = Process(target=run_flask, args=(port,))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

