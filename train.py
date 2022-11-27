import os
import torch
import random
import argparse
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from utils import io
from utils import plot
from utils import metric
from model import train_evaluate

from model import xlmr_xnli_model
from model import xlmr_xnli_dataset

from transformers import XLMRobertaTokenizer, XLMRobertaModel

from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--train_file', type=str, help="Training file.")
parser.add_argument('--batch_size', type=int, help="batch size.")
parser.add_argument('--device_name', type=str, help="Device to run the model on.",
                    default='cuda')


if __name__ == "__main__":

    args = parser.parse_args()

    device = args.device_name
    if device != 'cpu' and device[:4] != 'cuda':
        raise Exception(f"Invalid device name -- {device}")
    device = torch.device(device)
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    """
    Train dataset
    """
    if not os.path.exists(args.train_file):
        raise Exception(f"{args.train_file} not found")

    train_file = args.train_file
    train_ext = train_file[-3:]

    if train_ext == "tsv":
        data = io.load_xnli_dataset(train_file)
    elif train_ext == "csv":
        data = io.load_xnli_dataset_csv(train_file)
    else:
        raise Exception(f'Invalid training extension : {train_file}')


    """
    Language information
    """
    languages = ['zh', 'es', 'hi', 'sw']
    lang_code_map = {x:i for i, x in enumerate(data.language.cat.categories)}
    lang_codes = {lang_code_map[lang]: lang for lang in languages}

    dataset_info = {
            'language': data.language.cat.categories.values,
            'gold_labels': data.gold_label.cat.categories.values
    }


    """
    Train-test split
    """
    train_data, valid_data, test_data = io.split_dataset(data, lang_codes=lang_codes)


    """
    Dataset and dataloader
    """
    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
    train_dataset = xlmr_xnli_dataset.XLMRXNLIDataset(train_data, tokenizer, torch.device('cpu'))
    valid_dataset = xlmr_xnli_dataset.XLMRXNLIDataset(valid_data, tokenizer, torch.device('cpu'))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=0,
        shuffle=True,
        collate_fn=tokenizer.pad
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=0,
        shuffle=True,
        collate_fn=tokenizer.pad
    )


    """
    Model
    """
    model = XLMRobertaModel.from_pretrained("xlm-roberta-base")

    model_params = {'model': model,
                    'device': device,
                    'lstm_params':{
                        'input_size': model.config.hidden_size,
                        'hidden_size':model.config.hidden_size//2,
                        'num_layers': 2,
                        'batch_first':True,
                        'bidirectional':True,
                        'device':device
                    },
                    'attention_params':{
                        'embed_dim' : model.config.hidden_size,
                        'num_heads': 3,
                        'batch_first': True,
                        'device':device
                    },
                    'dropout_params':{
                        'xlmr_drop':0.5,
                        'lstm_drop':0.5,
                        'attn_drop':0.5,
                        'mlp_drop':0.5
                    },
                    'layers': [768, 3]
                    }
    xnli_model = xlmr_xnli_model.XLMRXLNIModel(**model_params)


    """
    Training
    """
    metric_params = {
        'accuracy': metric.accuracy,
        'macro_f1': metric.macro_f1,
        'average_f1': metric.average_f1,
    }

    train_params_base = {
        'num_epochs': 1,
        'step_size': 3,
        'gamma': 0.1,
        'lr': 1e-3,
        'betas': (0.9, 0.999),
        'lrs': [1e-5, 1e-3, 1e-3],
        'lang_codes': lang_codes,
        'weight_decay': 0,
        'save_dir':'experiments/LinearHead/',
        'save_tag':'',
        'verbose': True,
        'restore_file': None, #last, best
        'tensorboard_dir': 'runs/LinearHead/',
        'device': device
    }


    train_params = io.setup_training(train_params_base, model_params, dataset_info)
    xnli_model.freeze_layer()

    summary = train_evaluate.train_and_evaluate(xnli_model, train_dataloader,
                                                valid_dataloader, metric_params,
                                                train_params, continue_training=False)





