
import os
import pandas as pd
import numpy as np

import torch
from tqdm.notebook import tqdm

from transformers import BertTokenizer
from torch.utils.data import TensorDataset

from transformers import BertForSequenceClassification

model_path = r'../../../models/bert-base-uncased'
print(os.path.abspath(model_path))
model = BertForSequenceClassification.from_pretrained(model_path)

# In[16]:
#model = BertForSequenceClassification.from_pretrained(r'../bert-base-uncased', from_pt=True)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
model.save_pretrained(r'E:\Development\xxx\aaa.bin')
a = 1