import os
import ner_flaskr
#from ner_flaskr import app

if __name__ == '__main__':
    port = 5000
    url = '127.0.0.1'
    ner_flaskr.app.run(host=url, port=port)

    # from ner_flaskr.app import *