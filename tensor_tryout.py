import re
from time import time
from collections import Counter

import tensorflow as tf
import pandas as pd
import numpy as np

from nltk.stem.porter import PorterStemmer
from fastcache import clru_cache as lru_cache

# Data inladen
data = pd.read_table("../cleaned.csv")
print(data.iloc[0])


# Create dense network
# Kernel_intializer en kerner_regularizer snap ik nog niet echt
def dense(X, size, reg=0.0, activation=None):
    he_std = np.sqrt(2 / int(X.shape[1]))
    out = tf.layers.dense(X, units=size, activation=activation,
                     kernel_initializer=tf.random_normal_initializer(stddev=he_std),
                     kernel_regularizer=tf.contrib.layers.l2_regularizer(reg))
    return out


graph = tf.Graph()
graph.seed = 1

with graph.as_default():
# Learning rate
	place_lr = tf.placeholder(tf.float32, shape=(), )
	place_condition = tf.placeholder(tf.uint8, shape=(None, 1))

	place_y = tf.placeholder(dtype=tf.float32, shape=(None, 1))
#One hot conversion naar binair in extra rank. In plaats van getallen (1 tot 5)
	cond = tf.one_hot(place_condition, 5)
	cond = tf.contrib.layers.flatten(cond)
	print(cond.shape)


# Ik weet niet precies wat tf.nn.relu is
	cond = dense(cond, 100, activation=tf.nn.relu)
	cond = tf.layers.dropout(cond, rate=0.5)
	cond = dense(cond, 1)

	loss = tf.losses.mean_squared_error(place_y, cond)
	rmse = tf.sqrt(loss)
	opt = tf.train.AdamOptimizer(learning_rate=place_lr)
	train_step = opt.minimize(loss)

	init = tf.global_variables_initializer()

session = tf.Session(config=None, graph=graph)
session.run(init)

for i in range(4):
	for index in range(10):
		feed_dict = {
			place_condition: data.iloc[index, 3],
			place_y: data.iloc[index, 6],
		}
		lr = 0.0001
		session.run(train_step, feed_dict = feed_dict)

print("Training completed")



# #    t0 = time()
#     np.random.seed(i)
#     train_idx_shuffle = np.arange(X_name.shape[0])
#     np.random.shuffle(train_idx_shuffle)
#     batches = prepare_batches(train_idx_shuffle, 500)

#     if i <= 2:
#         lr = 0.001
#     else:
#         lr = 0.0001

#     for idx in batches:
#         feed_dict = {
#             place_name: X_name[idx],
#             place_desc: X_desc[idx],
#             place_brand: X_brand[idx],
#             place_cat: X_cat[idx],
#             place_cond: X_item_cond[idx],
#             place_ship: X_shipping[idx],
#             place_y: y[idx],
#             place_lr: lr,
#         }
#         session.run(train_step, feed_dict=feed_dict)

#     took = time() - t0
#     print('epoch %d took %.3fs' % (i, took))



# feed_dict = {
#             # place_name: X_name[idx],
#             # place_desc: X_desc[idx],
#             # place_brand: X_brand[idx],
#             # place_cat: X_cat[idx],
#             place_cond: X_item_cond[idx],
#             # place_ship: X_shipping[idx],
#             place_y: y[idx],
#             place_lr: lr,
#         }