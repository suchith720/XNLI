import os
import re
import json
import torch
import shutil
import logging
import numpy as np
import pandas as pd


def language_based_dataset_filter(dataset, lang_list):
    dataset_list = []
    for lang_code in lang_list:
        dataset_list.append(dataset[dataset.language == lang_code])
    dataset = pd.concat(dataset_list)
    dataset.reset_index(drop=True, inplace=True)
    return dataset


def load_xnli_dataset(filename, lang_list=None):
    dataset = pd.read_csv(filename, delimiter='\t',
                          dtype={'gold_label':'category', 'premise':str,
                                 'hypothesis':str, 'language':'category'})

    dataset['language'] = dataset.language.astype('category')

    vocab = ['contradiction', 'neutral', 'entailment']
    dataset['gold_label'] = pd.Categorical(dataset.gold_label,
                                           ordered=True,
                                           categories=vocab)
    dataset.dropna(axis=0, inplace=True)

    return dataset


def load_xnli_dataset_csv(filename, lang_list=None):
    dataset = pd.read_csv(filename, index_col=0,
                          dtype={'gold_label':'category', 'premise':str,
                                 'hypothesis':str, 'language':'category'})

    dataset['language'] = dataset.language.astype('category')

    vocab = ['contradiction', 'neutral', 'entailment']
    dataset['gold_label'] = pd.Categorical(dataset.gold_label,
                                           ordered=True,
                                           categories=vocab)
    dataset.dropna(axis=0, inplace=True)

    return dataset


def load_xnli_test_dataset(filename, languages):
    dataset = pd.read_csv(filename, delimiter='\t',
                          dtype={'premise':str, 'hypothesis':str,
                                 'language':'category'})
    dataset['language'] = pd.Categorical(dataset.language, ordered=True,
                                         categories=languages)
    return dataset

def save_xnli_test_dataset(test_input_file, test_output_file, test_data):
    test_input_df = test_data[['premise', 'hypothesis', 'language']]
    test_output_df = test_data['gold_label']

    test_input_df.to_csv(test_input_file, sep='\t', index=False)
    test_output_df.to_csv(test_output_file, sep='\t', index=False, header=False)



def split_dataset(df, pc=0.7, lang_codes=None):
    if lang_codes is None:
        lang_codes = df.language.unique()

    """
    Train-test split based on language
    """
    train_list, valid_list, test_list = list(), list(), list()
    for lang_code in lang_codes:
        lang_data = df[df.language.cat.codes == lang_code]

        num_rows = lang_data.shape[0]
        n_trn = int(num_rows * pc)
        val_len = int((1 - pc)/2 * num_rows)

        train_list.append(lang_data.iloc[:n_trn])
        valid_list.append(lang_data.iloc[n_trn: n_trn+val_len])
        test_list.append(lang_data.iloc[n_trn+val_len:])

    """
    Putting rest languages in training set.
    """
    all_lang_codes = df.language.cat.codes.unique()
    lang_codes = set(all_lang_codes).difference(lang_codes)
    for lang_code in lang_codes:
        lang_data = df[df.language.cat.codes == lang_code]
        train_list.append(lang_data)


    train_data = pd.concat(train_list).reset_index(drop=True)
    valid_data = pd.concat(valid_list).reset_index(drop=True)
    test_data = pd.concat(test_list).reset_index(drop=True)

    return train_data, valid_data, test_data



def readWord2Vector(word2VecFile):

    with open(word2VecFile, "r") as file:
        line = file.readline()
        vocab_size, word_dim = list(map(int, line[:-1].split(' ')))

        word2Vec, vocab  = [], {}
        for i, line in enumerate(file.readlines()):
            line = line[:-1].split()

            word = line[0]
            vocab[word] = i

            vec = list(map(float, line[1:]))
            word2Vec.append(vec)

    return word2Vec, vocab


def set_logger(log_path):

    if not os.path.exists(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        logger.addHandler(stream_handler)

    return logger


def reset_logger():
    logger = logging.getLogger()

    handlers = logger.handlers[:]
    for handler in handlers:
        logger.removeHandler(handler)
        handler.close()


def save_checkpoint(state, is_best, save_dir, tag=''):
    save_filename = f'{save_dir}/last{tag}.pth.tar'

    if not os.path.exists(save_dir):
        logging.info(f"'{save_dir}' does not exist, creating new directory.")
        os.makedirs(save_dir)

    torch.save(state, save_filename)
    if is_best:
        best_filename = f'{save_dir}/best{tag}.pth.tar'
        #best_filename = 'aiz218323_model'
        shutil.copyfile(save_filename, best_filename)


def load_checkpoint(save_file, model, device, optimizer=None):
    if not os.path.exists(save_file):
        raise Exception(f"'{save_file}' does not exist.")

    state = torch.load(save_file, map_location=device)
    model.load_state_dict(state['state_dict'])
    if optimizer:
        optimizer.load_state_dict(state['optim_dict'])

    return state


def save_dict_to_json(metrics, filename):
    dirname = os.path.dirname(filename)
    if not dirname:
        dirname = "."

    if not os.path.exists(dirname):
        raise Exception(f"'{dirname}' does not exist.")

    with open(filename, "w") as f:
        json.dump(metrics, f)


def load_dict_to_json(filename):
    if not os.path.exists(filename):
        raise Exception(f'{filename}, does not exist.')

    with open(filename, "r") as f:
        return json.load(f)


def save_info_file(save_file, model_params, dataset_info):
    save_dict = {
        'model_params': model_params,
        'dataset_info': dataset_info
    }
    torch.save(save_dict, save_file)

def load_info_file(load_file):
    save_dict = torch.load(load_file, map_location='cpu')
    return save_dict['model_params'], save_dict['dataset_info']


def setup_training(train_params, model_params, dataset_info):
    train_params = train_params.copy()

    os.makedirs(train_params['save_dir'], exist_ok=True)
    os.makedirs(train_params['tensorboard_dir'], exist_ok=True)

    num_runs = len(os.listdir(train_params['save_dir']))

    filename = f"{train_params['save_dir']}/R_{num_runs-1:03d}/"
    if os.path.exists(filename) and len(os.listdir(filename)) < 2:
        num_runs -= 1

    train_params['save_tag'] = f'_{num_runs:03d}'
    train_params['save_dir'] = f"{train_params['save_dir']}/R_{num_runs:03d}/"
    train_params['tensorboard_dir'] = f'{train_params["tensorboard_dir"]}/R_{num_runs:03d}'

    os.makedirs(train_params['save_dir'], exist_ok=True)
    os.makedirs(train_params['tensorboard_dir'], exist_ok=True)

    log_path = f"{train_params['save_dir']}/log.txt"
    logger = set_logger(log_path)

    save_file = f"{train_params['save_dir']}/info{train_params['save_tag']}.pth.tar"
    save_info_file(save_file, model_params, dataset_info)

    return train_params



