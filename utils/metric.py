import numpy as np
from sklearn.metrics import f1_score, classification_report


def languagewise_metrics(y_pred, y_true, lang, lang_codes):
    lang_dict = {}

    all_pred, all_true = [], []
    for lang_code, lang_str in lang_codes.items():
        mask = np.array(lang) == lang_code

        pred = np.array(y_pred)[mask].tolist()
        true = np.array(y_true)[mask].tolist()
        lang_dict[lang_str] = (pred, true)

        all_pred.extend(pred)
        all_true.extend(true)

    lang_dict['all'] = (all_pred, all_true)
    return lang_dict


def calculate_metric(y_pred, y_true, lang, lang_codes, average):
    lang_dict = languagewise_metrics(y_pred, y_true, lang, lang_codes)

    lang_metric = {}
    for lang_str, (y_pred, y_true) in lang_dict.items():
        lang_metric[lang_str] = f1_score(y_true, y_pred, average=average)
    return lang_metric



def accuracy(y_pred, y_true, lang, lang_codes):
    lang_metric = calculate_metric(y_pred, y_true, lang, lang_codes, 'micro')
    return lang_metric


def macro_f1(y_pred, y_true, lang, lang_codes):
    lang_metric = calculate_metric(y_pred, y_true, lang, lang_codes, 'macro')
    return lang_metric


def average_f1(y_pred, y_true, lang, lang_codes):
    acc_metric = accuracy(y_pred, y_true, lang, lang_codes)
    mac_metric = macro_f1(y_pred, y_true, lang, lang_codes)

    avg_metric = {}
    for lang_str in acc_metric:
        avg_metric[lang_str] = (acc_metric[lang_str] + mac_metric[lang_str])/2
    return avg_metric


def classification_summary(y_pred, y_true):
    return classification_report(y_true, y_pred, zero_division=True)

def update_avg_metrics(avg_metrics, metrics):
    for name in metrics:
        metric_vals = avg_metrics.setdefault(name, [])
        metric_vals.append( np.mean(metrics[name]) )

def update_metrics(metric_history, metrics):
    for name in metrics:
        metric_vals = metric_history.setdefault(name, [])
        metric_vals.extend( metrics[name] )

def update_metric_history(metric_history, lang_history, metrics):
    for name in metrics:
        if name == "loss":
            metric_vals = metric_history.setdefault(name, [])
            metric_vals.append( metrics[name] )
        else:
            for lang_str in metrics[name]:
                if lang_str == "all":
                    metric_vals = metric_history.setdefault(name, [])
                    metric_vals.append( metrics[name][lang_str] )
                else:
                    metric_vals = lang_history.setdefault(name, {})
                    lang_vals = metric_vals.setdefault(lang_str, [])
                    lang_vals.append( metrics[name][lang_str] )

