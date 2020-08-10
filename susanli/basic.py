from transformers import pipeline
unmasker = pipeline('fill-mask', model='distilbert-base-uncased')
unmasker("Hello I'm a [MASK] model.")