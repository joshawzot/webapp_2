# run.py

from werkzeug.serving import make_server  # can't use https yet, if want to use https, has to use server like Nginx or Apache
from my_flask_app import app

'''werkzeug: This is the name of the Python package.
This allows for building web applications and servers. It is designed to be a base for Python web applications or frameworks, 
providing a lot of code needed for handling HTTP requests and responses.
'''
'''
serving: This is a module within the Werkzeug package. 
The serving module provides simple utilities for running a WSGI application. 
It includes development server capabilities, which are useful for running a web application locally during the development process. 
Although primarily intended for development and testing, it provides essential functionalities for HTTP request handling.
'''
'''
make_server: This is a function within the werkzeug.serving module. 
The make_server function is used to create a new WSGI server instance. 
It allows you to specify the hostname, port, and the WSGI application that should be served. Optionally, 
you can enable or disable threading and choose other server configurations. 
Essentially, it's a way to instantiate a simple, yet flexible, WSGI-compliant web server that can serve your Flask (or any WSGI-compatible) application.
'''

def run_flask():
    #server = make_server('0.0.0.0', 5000, app)    #'0.0.0.0': The server is bound to all network interfaces, making it accessible from any IP address that can reach the host machine.
    #server = make_server('localhost', 8000, app)  # This tells the server to bind to the localhost network interface, which is accessible only from the same machine on which the server is running.
                                                   #A WSGI server is created to serve the app (a Flask application, in this case) on localhost at port 8000.
    server = make_server('0.0.0.0', 5000, app, threaded=True)  # Enable multi-threading here
    server.serve_forever()

'''It provides a quick and straightforward way to get your web application up and running for testing purposes. 
However, for production environments, it's recommended to use more robust WSGI servers like Gunicorn or uWSGI, 
as they offer better performance, security, and scalability.
'''

if __name__ == "__main__":  #in app.py, there is "app = Flask(__name__) "
    # This block of code will only execute when the script is run directly.
    run_flask()


#5000: The port number on which the server will listen for incoming connections.
#threaded=True: Enables multi-threading, allowing the server to handle multiple requests concurrently.
#server.serve_forever(): Starts the server, making it continuously listen for and respond to incoming requests until it is manually stopped.

#When you directly run a Python script, Python sets __name__ to "__main__" to indicate that this script is the main program.

'''
# module_example.py

def greet(name):
    print(f"Hello, {name}!")

if __name__ == "__main__":
    print("This prints only if module_example.py is executed directly.")
    greet("Alice")

When you run module_example.py directly (e.g., with python module_example.py), it outputs:
This prints only if module_example.py is executed directly.
Hello, Alice!



# import_example.py

import module_example

module_example.greet("Bob")

When you run import_example.py (e.g., with python import_example.py), it outputs:
Hello, Bob!

This is because when module_example.py is imported into import_example.py, 
the __name__ variable in module_example.py is set to "module_example" (the name of the module), not "__main__". 
Therefore, the condition if __name__ == "__main__": evaluates to False, and the code block under it is not executed.
'''