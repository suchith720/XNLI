import os
import torch
import logging
import tensorboardX
from torch import nn
from tqdm.notebook import tqdm
import torch.nn.functional as F
from timeit import default_timer as timer

from utils.io import *
from utils.plot import *
from utils.metric import *


def train(model, dataloader, criterion, optimizer, metrics, lang_codes, device):
    model.train()

    train_metrics = {}

    loss_history = []
    y_preds, y_trues, langs = [], [], []
    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        batch = batch.to(device)
        optimizer.zero_grad()

        y_pred = model(batch)
        y_true = batch['labels']
        lang = batch['language']

        loss = criterion(y_pred, y_true)
        loss.backward()

        optimizer.step()

        loss_history.append(loss.item())

        y_preds.extend(y_pred.data.cpu().argmax(dim=1).tolist())
        y_trues.extend(y_true.data.cpu().tolist())
        langs.extend(lang.data.cpu().tolist())

    train_metrics['loss'] = np.mean(loss_history)
    for name, metric in metrics.items():
        train_metrics[name] = metric(y_preds, y_trues, langs, lang_codes)

    return train_metrics


def evaluate(model, dataloader, criterion, metrics, lang_codes, device):
    model.eval()

    valid_metrics = {}

    loss_history = []
    y_preds, y_trues, langs = [], [], []
    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        batch = batch.to(device)

        y_pred = model(batch)
        y_true = batch['labels']
        lang = batch['language']

        loss = criterion(y_pred, y_true)

        loss_history.append(loss.item())

        y_preds.extend(y_pred.data.cpu().argmax(dim=1).tolist())
        y_trues.extend(y_true.data.cpu().tolist())
        langs.extend(lang.data.cpu().tolist())

    valid_metrics['loss'] = np.mean(loss_history)
    for name, metric in metrics.items():
        valid_metrics[name] = metric(y_preds, y_trues, langs, lang_codes)

    return valid_metrics


def predict(model, dataloader, vocab, device):
    model.eval()

    labels = []
    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        batch = batch.to(device)
        y_pred = model(batch)
        labels.extend(y_pred.data.cpu().argmax(dim=1).tolist())

    return np.array(vocab)[labels].tolist()


def train_and_evaluate(model, train_dataloader, valid_dataloader,
                       metric_params, train_params, continue_training=False):

    criterion = torch.nn.NLLLoss()


    params = [ {'params': param.parameters(), 'lr': lr}
              for param, lr in zip(model.params.values(), train_params['lrs'])]
    optimizer = torch.optim.Adam(params,
                                 lr=train_params['lr'],
                                 weight_decay=train_params['weight_decay'])


    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, train_params['step_size'],
                                                gamma=train_params['gamma'])


    if train_params['restore_file'] is not None:
        load_checkpoint(train_params['restore_file'], model, train_params['device'],
                        optimizer=optimizer)


    #layout={
    #        "LOSSES":{
    #            "loss": ["Multiline", ["loss/train", "loss/valid"]],
    #            "accuracy": ["Multiline", ["accuracy/train", "accuracy/valid"]],
    #            "macro_f1": ["Multiline", ["macro_f1/train", "macro_f1/valid"]],
    #            "average_f1": ["Multiline", ["average_f1/train", "average_f1/valid"]],
    #        },
    #        }

    #if train_params['tensorboard_dir'] is None:
    #    writer = tensorboardX.SummaryWriter()
    #else:
    #    if not os.path.exists(train_params['tensorboard_dir']):
    #        logging.info(f"'{train_params['tensorboard_dir']}' does not exist.")
    #        os.makedirs(train_params['tensorboard_dir'])
    #    writer = tensorboardX.SummaryWriter(train_params['tensorboard_dir'])
    #writer.add_custom_scalars(layout)


    train_time, valid_time = list(), list()
    train_history, valid_history = dict(), dict()
    train_lang_history, valid_lang_history = dict(), dict()


    best_acc_value = 0.0
    if continue_training:
        best_file = f'{train_params["save_dir"]}/best_acc_value{train_params["save_tag"]}.json'
        current_best_acc_value = load_dict_to_json(best_file)
        best_acc_value = current_best_acc_value['best_acc_value']


    #best_valid_metrics = {}
    for epoch in tqdm(range(train_params['num_epochs']), total=train_params['num_epochs']):

        logging.info(f"Epoch {epoch+1}/{train_params['num_epochs']}")

        epoch_start = timer()
        train_metrics = train(model, train_dataloader, criterion, optimizer, metric_params,
                              train_params['lang_codes'], train_params['device'])
        #for k, v in train_metrics.items():
        #    writer.add_scalar(f'{k}/train', v, epoch+1)
        end_epoch = timer()
        train_time.append(end_epoch - epoch_start)
        update_metric_history(train_history, train_lang_history, train_metrics)


        epoch_start = timer()
        valid_metrics = evaluate(model, valid_dataloader, criterion, metric_params,
                                 train_params['lang_codes'], train_params['device'])
        #for k, v in valid_metrics.items():
        #    writer.add_scalar(f'{k}/valid', v, epoch+1)
        end_epoch = timer()
        valid_time.append(end_epoch - epoch_start)
        update_metric_history(valid_history, valid_lang_history, valid_metrics)


        """
        scheduler step
        """
        scheduler.step()


        """
        Saving the state.
        """
        state = {"epoch":epoch+1, "state_dict":model.state_dict(), "optim_dict":optimizer.state_dict()}
        is_best = valid_history['average_f1'][-1] > best_acc_value
        save_checkpoint(state, is_best, train_params["save_dir"], train_params['save_tag'])


        epoch_metric = pretty_epoch_metrics(epoch+1, train_history, valid_history, train_time, valid_time)
        lang_metric = pretty_metrics(epoch+1, train_lang_history, valid_lang_history)

        if is_best:
            logging.info("- Found new best accuracy.")

            best_acc_value = valid_history['average_f1'][-1]
            #best_valid_metrics = {"hparam/accuracy":best_acc_value,
            #                      "hparam/loss":valid_history['loss'][-1],
            #                      "hparam/f1":valid_history['macro_f1'][-1],
            #                      "hparam/average_f1":valid_history['average_f1'][-1],
            #                      }

            metric_file = f'{train_params["save_dir"]}/best_metrics{train_params["save_tag"]}.json'
            save_dict_to_json(epoch_metric, metric_file)

            """
            logging current best accuracy to continue training
            """
            current_best_acc_value = {'best_acc_value':best_acc_value}
            best_file = f'{train_params["save_dir"]}/best_acc_value{train_params["save_tag"]}.json'
            save_dict_to_json(current_best_acc_value, best_file)

        metric_file = f'{train_params["save_dir"]}/last_metrics{train_params["save_tag"]}.json'
        save_dict_to_json(epoch_metric, metric_file)


        """
        Display metrics
        """
        if train_params['verbose']:
            display(pd.DataFrame(display_metrics(epoch_metric)).T)
            for lm in lang_metric:
                display(pd.DataFrame(display_metrics(lm)).T)



    summary = {
        'train_history': train_history,
        'valid_history': valid_history,
    }
    logging.info(f'- Total training time : {np.sum(train_time) + np.sum(valid_time)} secs')
    #writer.close()
    return summary

