import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class M2M100Dataset(Dataset):

    def __init__(self, data_dict, lang_code, tokenizer, device):
        self.device = device
        self.lang_code = lang_code

        data = data_dict[lang_code]

        input_pairs = list(zip(data['premise'], data['hypothesis']))

        tokenizer.src_lang = lang_code

        self.encodings = dict()
        self.encodings['premise'] = tokenizer.batch_encode_plus(data['premise'], padding=True, truncation=True)
        self.encodings['hypothesis'] = tokenizer.batch_encode_plus(data['hypothesis'], padding=True, truncation=True)
        self.encodings['label'] = [torch.tensor(x, dtype=torch.int64, device=self.device) for x in data['gold_label']]
        self.convert_to_tensor(self.encodings['premise'])
        self.convert_to_tensor(self.encodings['hypothesis'])

    def convert_to_tensor(self, data):
        data['input_ids'] = [torch.tensor(x, dtype=torch.int64, device=self.device) for x in data['input_ids']]
        data['attention_mask'] = [torch.tensor(x, dtype=torch.int64, device=self.device) for x in data['attention_mask']]

    def __len__(self):
        return len(self.encodings['label'])

    def __getitem__(self, idx):
        return {'label': self.encodings['label'][idx],
                'premise': {k:self.encodings['premise'][k][idx] for k in self.encodings['premise'].keys()},
                'hypothesis': {k:self.encodings['hypothesis'][k][idx] for k in self.encodings['hypothesis'].keys()}
                }

