import os
from transformers import pipeline

model_path = r'../../../models/bert-base-uncased'
print(os.path.abspath(model_path))

unmasker = pipeline('fill-mask', model='distilbert-base-uncased')
unmasker = pipeline('fill-mask', model=model_path)
print(unmasker("Hello I'm a [MASK] model."))
print('Done')

