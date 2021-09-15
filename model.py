import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import pysnooper

train_data = pd.read_csv('./train_data.csv')
data_x = train_data.loc[:, [a for a in train_data.columns if a not in ['time_id', 'target', 'stock_id']]]
data_y = train_data.loc[:, train_data.columns == 'target']
data_y['target'].astype('float64')


train_x, train_y = data_x[:400000], data_y[:400000]
test_x, test_y = data_x[400000:], data_y[400000:]

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(11,)),
  tf.keras.layers.Dense(10, activation='relu'),
  tf.keras.layers.Dense(7, activation='relu'),
  # tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(5, activation='tanh'),
  tf.keras.layers.Dense(3, activation='tanh'),
  tf.keras.layers.Dense(1, activation='tanh')
])

adam = keras.optimizers.Adam(lr=0.0000001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=adam,
              loss='mean_squared_error')

model.fit(train_x, train_y, epochs=100)
model.evaluate(test_x, test_y)

pred = model.predict(test_x)
print(pred)

