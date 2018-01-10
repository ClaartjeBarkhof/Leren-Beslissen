
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

def error_function(labels, predicted):
    # Y and Y_red have already been in log scale.
    assert Y.shape == Y_pred.shape
    return np.sqrt(np.mean(np.square(Y_pred - Y )))

# Load the data
input_train = pd.read_table("../train.tsv")
input_train_txt = input_train.as_matrix()
target_train = input_train['price']
print("Data is loaded")
print(input_train[0:1])

# Basic regression (taken from http://scikit-learn.org/stable/modules/linear_model.html)
reg = linear_model.LinearRegression()
reg.fit(input_train, target_train)

