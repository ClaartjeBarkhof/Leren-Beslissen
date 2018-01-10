
import numpy as np
import pandas as pd

from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dropout, Dense, concatenate, GRU, Embedding, Flatten, Activation
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K


# Packages van "Tensorflow starter conv1d"
# import re
# from collections import Counter
# import tensorflow as tf
# from nltk.stem.porter import PorterStemmer
# from fastcache import clru_cache as lru_cache
# from time import time

def error_function(labels, predicted):
    # Y and Y_red have already been in log scale.
    assert labels.shape == predicted.shape
    return np.sqrt(np.mean(np.square(predicted - labels)))

# Load the data
input_train = pd.read_table("../train.tsv")
input_train_txt = input_train.as_matrix()
target_train = input_train['price']
print("Data is loaded")

# Returns encoded discrete values of a single column
def label_encoder(column):
	le = LabelEncoder()
	le.fit(column)
	encoded_column = le.transform(column)
	del le
	return encoded_column

# Testing the label encoder
print (input_train['category_name'].iloc[0:3])
print label_encoder(input_train['category_name'].iloc[0:3])

# Basic regression (taken from http://scikit-learn.org/stable/modules/linear_model.html)
reg = linear_model.LinearRegression()
reg.fit(input_train, target_train)
