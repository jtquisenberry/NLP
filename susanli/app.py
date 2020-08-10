import ktrain
import pandas as pd

print(ktrain.version)
print(ktrain.__version__)

import torch
from tqdm.notebook import tqdm

from transformers import BertTokenizer
from torch.utils.data import TensorDataset

from transformers import BertForSequenceClassification



print(torch.__version__)
df = pd.read_csv('data/title_conference.csv')

a = 1