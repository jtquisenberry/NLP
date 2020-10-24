from datetime import datetime
from flask import request, url_for, Flask, render_template, request, redirect, current_app, g, send_from_directory, session
import json
import os
from transformers import TFAutoModelForTokenClassification, AutoTokenizer
import tensorflow as tf

# Ignoring visible gpu device (device: 0, name: GeForce GTX 660 Ti, pci bus id: 0000:05:00.0,
# compute capability: 3.0) with Cuda compute capability 3.0. The minimum required Cuda capability is 3.5.
# Force use of CPU even if a GPU is available.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")
print()

model = TFAutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


__version__ = '1.00'
from flask import Flask
app = Flask(__name__)
app.debug = True

with app.app_context():

    context_test = ['testing context']

    app.config['model'] = model
    app.config['tokenizer'] = tokenizer

    from controllers import ner
    ner.check_loaded()

q = 1