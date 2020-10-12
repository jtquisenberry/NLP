import os

# Ignoring visible gpu device (device: 0, name: GeForce GTX 660 Ti, pci bus id: 0000:05:00.0,
# compute capability: 3.0) with Cuda compute capability 3.0. The minimum required Cuda capability is 3.5.
# Force use of CPU even if a GPU is available.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import sys
import re
from transformers import TFAutoModelForTokenClassification, AutoTokenizer
import tensorflow as tf

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")
print()

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

os.chdir(os.path.dirname(os.path.abspath(__file__)))
text_file = r'./princess_bride.txt'
with open(text_file, encoding='utf-8') as file:
    sequence = file.read()

# A method to get the tokens with the special tokens
#tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sequence)))

re_words = re.findall(r'[a-zA-Z0-9]+', sequence)
print('Regular Expression Words: {0}'.format(len(re_words)))

tokens = tokenizer.tokenize(sequence)
print('BERT Subtokens: {0}'. format(len(tokens)))

# A method to get the tokens with the special tokens
tokens2 = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sequence)))
print('BERT Subtokens 2: {0}'. format(len(tokens2)))

a = 1




a = 1