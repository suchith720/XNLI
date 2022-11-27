import os
import torch
import random
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from utils import io
from utils import plot
from utils import metric
from model import train_evaluate
from model import xlmr_xnli_model
from model import xlmr_xnli_dataset

from transformers import XLMRobertaTokenizer, XLMRobertaModel


parser = argparse.ArgumentParser()

parser.add_argument('--input_file', type=str, help="Test file to do prediction on.")
parser.add_argument('--output_file', type=str, help="output file")

parser.add_argument('--model_file', type=str, help="file with model weights.")
parser.add_argument('--info_file', type=str, help="information file containing model parameters details")

parser.add_argument('--batch_size', type=int, help="information file containing model parameters details",
                    default=32)
parser.add_argument('--device_name', type=str, help="Device to run the model on.",
                    default='cpu')



if __name__ == "__main__":

    args = parser.parse_args()

    device = args.device_name
    if device != 'cpu' and device[:4] != 'cuda':
        raise Exception(f"Invalid device name -- {device}")
    device = torch.device(device)
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    """
    Load information file
    """
    info_file = args.info_file
    if not os.path.exists(info_file):
        raise Exception(f"{info_file} not found")
    model_params, dataset_info = io.load_info_file(info_file)


    """
    Load data
    """
    languages = ['zh', 'es', 'hi', 'sw']
    lang_code_map = {x:i for i, x in enumerate(dataset_info['language'])}
    lang_codes = [lang_code_map[lang] for lang in languages]

    data = io.load_xnli_test_dataset(args.input_file, dataset_info['language'])


    """
    Dataloader
    """
    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
    test_dataset = xlmr_xnli_dataset.XLMRXNLIDataset(data, tokenizer, torch.device('cpu'), isTrain=False)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=0,
        shuffle=False,
        collate_fn=tokenizer.pad
    )


    """
    Model
    """
    xnli_model = xlmr_xnli_model.XLMRXLNIModel(**model_params)
    state = io.load_checkpoint(args.model_file, xnli_model, device)
    output = train_evaluate.predict(xnli_model, test_dataloader,
                                    dataset_info['gold_labels'], device)


    """
    Save output
    """
    Path(os.path.dirname(args.output_file)).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output).to_csv(args.output_file, header=False, index=False)



