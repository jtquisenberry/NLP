# https://huggingface.co/transformers/usage.html


import os

# Ignoring visible gpu device (device: 0, name: GeForce GTX 660 Ti, pci bus id: 0000:05:00.0,
# compute capability: 3.0) with Cuda compute capability 3.0. The minimum required Cuda capability is 3.5.
# Force use of CPU even if a GPU is available.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from transformers import TFAutoModelForTokenClassification, AutoTokenizer
import tensorflow as tf

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")
print()


model = TFAutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

label_list = [
    "O",       # Outside of a named entity
    "B-MISC",  # Beginning of a miscellaneous entity right after another miscellaneous entity
    "I-MISC",  # Miscellaneous entity
    "B-PER",   # Beginning of a person's name right after another person's name
    "I-PER",   # Person's name
    "B-ORG",   # Beginning of an organisation right after another organisation
    "I-ORG",   # Organisation
    "B-LOC",   # Beginning of a location right after another location
    "I-LOC"    # Location
]




def get_ner(sequence):


    # Bit of a hack to get the tokens with the special tokens
    tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sequence)))
    inputs = tokenizer.encode(sequence, return_tensors="tf")

    outputs = model(inputs)[0]
    predictions = tf.argmax(outputs, axis=2)

    print([(token, label_list[prediction]) for token, prediction in zip(tokens, predictions[0].numpy())])

sequence = "Hugging Face Inc. is a company based in New York City. Its headquarters are in DUMBO, therefore very" \
           "close to the Manhattan Bridge."


get_ner(sequence=sequence)