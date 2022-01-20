import numpy as np
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine
from gen_name import gen_name

def gen_df(length, col_length=None, seed=None, att_chance=None, whitelist=None, extended_mode=0):
    """

    :param length:
    :param col_length:
    :param seed:
    :param att_chance: chance with which each attribute gets included. smaller means less likely/less attributes.
    :param whitelist:
    :param extended_mode: int from 0 to 3.
    0: simplified mode only. Subgraphs are only calculated for the specified degree.
    1: extended mode only. Subgraphs are calculated as in the paper with degrees up to and including the specified.
    2: random. Randomly decides between 0 and 1.
    3: both. Each hyperparam combination will be included with mode 1 and mode 2.
    Default is 0.
    :return:
    """
    length = length if length else 10
    rng = np.random.default_rng(seed) if seed else np.random.default_rng()  # potentially place seed here
    if extended_mode == 3:
        a = np.ones(length, dtype=int)
        b = np.zeros(length, dtype=int)
        extended_mode_param = np.concatenate((a,b), axis=None)
    elif extended_mode == 2:
        extended_mode_param = rng.integers(low=0, high=2, size=length)
    elif extended_mode == 1:
        extended_mode_param = np.ones(length, dtype=int)
    else: #extended_mode = 0
        extended_mode_param = np.zeros(length, dtype=int)

    att_chance = 50 if att_chance is None else att_chance

    epochs = rng.integers(low=1, high=7, size=length)*10
    neg_sam = 5 * rng.integers(low=2, high=6, size=length)
    vector_dim = 2 ** rng.integers(low=0, high=9, size=length)
    degree = rng.integers(low=0, high=5, size=length)
    window = rng.integers(low=1, high=5, size=length)
    min_count = rng.integers(low=1, high=5, size=length)
    names = [gen_name(2) for i in range(length)]

    alpha = rng.random(size=length) / 20
    alpha_digits = rng.integers(low=2,high =4, size=length)
    alpha = np.array([round(a,alpha_digits[index]) for index, a in enumerate(alpha)])
    alpha[alpha == 0] = 0.001 #if rounded to 0 set to smallest allowed value, 0.001

    unmasked_cols = ['dummy']*length # quick  + dirty fix
    df = pd.DataFrame(
        {'epochs': epochs, 'negative_sampling': neg_sam, 'vector_dimensions': vector_dim, 'degree': degree,
         'alpha': alpha, 'window': window, 'min_count':min_count, 'columns': unmasked_cols, 'name': names},
        columns=['epochs', 'negative_sampling', 'vector_dimensions', 'degree', 'alpha', 'window', 'min_count', 'columns','name'])

    if extended_mode_param.size == 2*length:
        #mode 3 detected
        #print('debug extended mode 3: ', extended_mode_param)
        df = pd.concat([df]*2, ignore_index=True)
        df['name'] = [gen_name(2) for i in range(2 * length)]

    df['extended_mode'] = extended_mode_param

    df = df.sort_values(by=['degree'], ascending=True)
    df = df.reset_index(drop=True)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    return df