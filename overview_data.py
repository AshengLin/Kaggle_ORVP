import numpy as np
import pandas as pd
import os


def data_trans(id):
    for dirname, _, filenames in os.walk('../data/book_train.parquet/stock_id='+str(id)):
        for filename in filenames:
            path_1 = os.path.join(dirname, filename)

    for dirname, _, filenames in os.walk('../data/trade_train.parquet/stock_id='+str(id)):
        for filename in filenames:
            path_2 = os.path.join(dirname, filename)

    book_train_1 = pd.read_parquet(path_1, engine='pyarrow')
    trade_train_1 = pd.read_parquet(path_2, engine='pyarrow')

    book_train = book_train_1.groupby(['time_id']).agg(avg_bid_price1=('bid_price1', 'mean'),
                                                       avg_ask_price1=('ask_price1', 'mean'),
                                                       avg_bid_price2=('bid_price2', 'mean'),
                                                       avg_ask_price2=('ask_price2', 'mean'),
                                                       sum_bid_size1=('bid_size1', 'sum'),
                                                       sum_ask_size1=('ask_size1', 'sum'),
                                                       sum_bid_size2=('bid_size1', 'sum'),
                                                       sum_ask_size2=('ask_size1', 'sum'))
    book_train['stock_id'] = str(id) + '_' + book_train.index.astype(str)
    trade_train = trade_train_1.groupby(['time_id']).agg(avg_price=('price', 'mean'),
                                                         sum_size=('size', 'sum'),
                                                         sum_order_count=('order_count', 'sum'))
    trade_train['stock_id'] = str(id) + '_' + trade_train.index.astype(str)
    feature = pd.merge(book_train, trade_train, on='stock_id')
    return feature


train_data = pd.read_csv('../data/train.csv')
stock = np.unique(train_data.stock_id)

for i in stock:
    print(data_trans(i))