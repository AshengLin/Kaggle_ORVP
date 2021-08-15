import tensorflow as tf
from tensorflow import keras

f1 = keras.models.Sequential([keras.layers.Dense(10, input_dim=1)], name="f1")