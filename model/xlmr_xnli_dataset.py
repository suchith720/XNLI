import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class XLMRXNLIDataset(Dataset):

    def __init__(self, data, tokenizer, device, isTrain=True):
        self.device = device
        self.isTrain = isTrain

        input_pairs = list(zip(data['premise'], data['hypothesis']))
        self.encodings = tokenizer.batch_encode_plus(input_pairs, truncation=True)#, padding=True,)
        self.encodings['language'] = data.language.cat.codes.tolist()
        if self.isTrain:
            self.encodings['labels'] = data.gold_label.cat.codes.tolist()

        self.convert_to_tensor()

    def convert_to_tensor(self):
        self.encodings['input_ids'] = [torch.tensor(x, dtype=torch.int64, device=self.device) for x in self.encodings['input_ids']]
        self.encodings['attention_mask'] = [torch.tensor(x, dtype=torch.int64, device=self.device) for x in self.encodings['attention_mask']]
        self.encodings['language'] = [torch.tensor(x, dtype=torch.int64, device=self.device) for x in self.encodings['language']]
        if self.isTrain:
            self.encodings['labels'] = [torch.tensor(x, dtype=torch.int64, device=self.device) for x in self.encodings['labels']]

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        return {k:self.encodings[k][idx] for k in self.encodings.keys()}


