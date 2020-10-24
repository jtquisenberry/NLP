import os
import project
#from project import app

if __name__ == '__main__':
    port = 5000
    url = '127.0.0.1'
    project.app.run(host=url, port=port)

    # from project.app import *