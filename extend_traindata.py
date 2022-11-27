import torch
import argparse
import pandas as pd

from utils import io
from utils import translate
from model import m2m100_dataset

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--train_file', type=str, help="training file")
parser.add_argument('--batch_size', type=int, help="batch size", default=16)
parser.add_argument('--save_dir', type=str, help="save directory")


if __name__ == "__main__":

    args = parser.parse_args()

    train_file = args.train_file
    save_dir = args.save_dir
    batch_size = args.batch_size

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    """
    Load data
    """
    data, vocab = io.load_xnli_dataset(train_file)
    data_dict = translate.create_languagewise_dict(data)

    """
    Model
    """
    model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
    model = model.to(device)

    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")


    """
    Translate text
    """
    log_path = f"{save_dir}/log.txt"
    logger = io.set_logger(log_path)

    trans_text = translate.TranslateText(model, tokenizer, data_dict, vocab, batch_size=batch_size, device=device)

    #lang_list = ['en', 'hi', 'sw', 'es', 'zh']
    #lang_list = ['en', 'vi', 'de', 'ar','bg', 'ur']
    lang_list = ['en', 'el', 'th', 'ru', 'tr', 'fr']

    save_file = save_dir + '/extended_train_{}.csv'
    trans_text.run(lang_list, save_file)

