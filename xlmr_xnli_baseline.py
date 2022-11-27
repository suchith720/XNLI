import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from utils import io
from utils import metric
from model import xlmr_xnli_dataset


def evaluate(model, dataloader, metrics):
    model.eval()

    valid_metrics = {}

    loss_history = []
    y_preds, y_trues = [], []
    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        output = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['label'])

        y_pred = output.logits
        y_true = batch['label']

        loss_history.append(output.loss.item())

        y_preds.extend(y_pred.data.cpu().argmax(dim=1).tolist())
        y_trues.extend(y_true.data.cpu().tolist())

    valid_metrics['loss'] = np.mean(loss_history)
    for name, metric in metrics.items():
        valid_metrics[name] = metric(y_preds, y_trues)

    return valid_metrics

parser = argparse.ArgumentParser()
parser.add_argument('--train_file', type=str, help="training file")
parser.add_argument('--batch_size', type=int, help="batch size.", default=256)

if __name__ == "__main__":

    args = parser.parse_args()

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    """
    Load Dataset
    """
    data, vocab = io.load_xnli_dataset(args.train_file)
    data = io.language_based_dataset_filter(data, ['hi', 'sw', 'zh', 'es'])

    """
    Model
    """
    tokenizer = AutoTokenizer.from_pretrained("joeddav/xlm-roberta-large-xnli")
    model = AutoModelForSequenceClassification.from_pretrained("joeddav/xlm-roberta-large-xnli").to(device)

    """
    Dataloader
    """
    dataset = xlmr_xnli_dataset.XLMRXNLIDataset(data, tokenizer, device)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,
    )

    """
    Evaluation
    """
    metric_params = {
            'accuracy': metric.accuracy,
            'macro_f1': metric.macro_f1,
            'average_f1': metric.average_f1

    }
    valid_metrics = evaluate(model, dataloader, metric_params)
    print(valid_metrics)


