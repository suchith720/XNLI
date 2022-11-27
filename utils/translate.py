import torch
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import io
from utils import translate
from model import m2m100_dataset

from torch.utils.data import DataLoader


def create_languagewise_dict(df):
    langwise_dict = dict()
    for lang_code in df.language.unique():
        mask = df['language'] == lang_code
        lang_df = df[mask]

        lang_dict = {}
        lang_dict['premise'] = lang_df['premise'].tolist()
        lang_dict['hypothesis'] = lang_df['hypothesis'].tolist()
        lang_dict['gold_label'] = lang_df['gold_label'].cat.codes.tolist()

        langwise_dict[lang_code] = lang_dict
    return langwise_dict


def create_language_df(langwise_dict):
    df_list = []

    for lang_code in langwise_dict.keys():
        df = pd.DataFrame(langwise_dict[lang_code])
        df['language'] = lang_code
        df_list.append(df)

    return pd.concat(df_list).reset_index(drop=True)


class TranslateText():

    def __init__(self, model, tokenizer, data_dict, vocab, batch_size, device):
        self.model = model.to(device)
        self.tokenizer = tokenizer

        self.vocab = vocab
        self.data_dict = data_dict

        self.batch_size = batch_size

        self.device = device


    def run(self, lang_list, save_file):

        for src_code in tqdm(lang_list, total=len(lang_list)):
            for tar_code in lang_list:

                if src_code != tar_code:
                    logging.info(f'Translating {src_code} to {tar_code}')
                    output = self.translate(src_code, tar_code)
                    df = pd.DataFrame(output)
                    df.to_csv(save_file.format(f'{src_code}_{tar_code}'))


    def translate(self, src_code, tar_code, max_new_tokens=500):
        dataset = m2m100_dataset.M2M100Dataset(self.data_dict, src_code,
                                               self.tokenizer, self.device)

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=0,
            shuffle=False,
        )

        tar_texts = {}
        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            outputs = self.model.generate(**batch['premise'], forced_bos_token_id=self.tokenizer.get_lang_id(tar_code),
                                      max_new_tokens=max_new_tokens)
            tar_premise = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            l = tar_texts.setdefault('premise', [])
            l.extend(tar_premise)

            outputs = self.model.generate(**batch['hypothesis'], forced_bos_token_id=self.tokenizer.get_lang_id(tar_code),
                                      max_new_tokens=max_new_tokens)
            tar_hypothesis = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            l = tar_texts.setdefault('hypothesis', [])
            l.extend(tar_hypothesis)

            l = tar_texts.setdefault('label', [])
            l.extend(np.array(self.vocab)[batch['label'].tolist()].tolist())

        return tar_texts

