import numpy as np
import pandas as pd

def filter_data(df, languages):
    lang_list = []
    for l in languages:
        lang_list.append(df[df['language'] == l])

    return lang_list

def display_random_text(lang_list):
    for l in lang_list:
        p = np.random.randint(len(l))
        x = l.iloc[p]

        for i in range(len(x)):
            print(x[i])
        print()

