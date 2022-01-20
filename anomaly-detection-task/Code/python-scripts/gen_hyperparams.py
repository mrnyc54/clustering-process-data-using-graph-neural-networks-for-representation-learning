from gen_name import gen_name
import numpy as np
import pandas as pd

def gen_df(length, seed=None):

    length = length if length else 10
    rng = np.random.default_rng(seed) if seed else np.random.default_rng()  # potentially place seed here

    epochs = rng.integers(low=1, high=30, size=length)*10

    hidden_dim_gin = 2 ** rng.integers(low=5, high=11, size=length) # dimension of the hidden layers of the GINConv model

    ratio_topk = rng.integers(low=1, high=10, size=length) / 10 # ratio of the topk pooling layers in TopKPooling model

    names = [gen_name(2) for i in range(length)]

    alpha = rng.random(size=length) / 20
    alpha_digits = rng.integers(low=2,high =4, size=length)
    alpha = np.array([round(a,alpha_digits[index]) for index, a in enumerate(alpha)])
    alpha[alpha == 0] = 0.001 #if rounded to 0 set to smallest allowed value, 0.001

    batch_size = 2 ** rng.integers(low=4, high=11, size=length)

    df = pd.DataFrame(
        {'epochs': epochs, 'alpha': alpha, 'batch_size':batch_size, 'hidden_gin':hidden_dim_gin, 'ratio_topk':ratio_topk, 'name': names},
        columns=['epochs', 'alpha', 'batch_size', 'hidden_gin', 'ratio_topk', 'name'])

    df = df.reset_index(drop=True)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    return df