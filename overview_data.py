import numpy as np
import pandas as pd
import os

for dirname, _, filenames in os.walk('../data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_data = pd.read_csv('../data/train.csv')
print(train_data)

book_train_1 = pd.read_parquet('../data/book_train.parquet/stock_id=0/c439ef22282f412ba39e9137a3fdabac.parquet', engine='pyarrow')
trade_train_1 = pd.read_parquet('../data/trade_train.parquet/stock_id=0/ef805fd82ff54fadb363094e3b122ab9.parquet', engine='pyarrow')

print(book_train_1)
print(trade_train_1)