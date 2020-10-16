# Fine-tuning BERT-based NER

Based on huggingface.

https://huggingface.co/transformers/v2.4.0/examples.html#named-entity-recognition


## run_ner
This is the original PyTorch version.

## ner_pt_runner
This is the PyTorch version wrapped in a class.

## run_tf_ner
This is the original TensorFlow version. IT DOES NOT WORK. 
``` python
tensorflow.python.framework.errors_impl.InvalidArgumentError:  Received a label value of -100 which is outside the valid range of [0, 9).
```

## ner_tf_runner
This is the TensorFlow version wrapped in a class.