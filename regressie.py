
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

# Load the data
input_train = pd.read_table("../train.tsv")
input_train_txt = input_train.as_matrix()
target_train = input_train['price']
input_test = pd.read_table("../test.tsv")

print("Data is loaded")

def error_function(labels, predicted):
    # Y and Y_red have already been in log scale.
    assert labels.shape == predicted.shape
    return np.sqrt(np.mean(np.square(predicted - labels)))

# Returns encoded discrete values of a single column
def label_encoder(column):
	le = LabelEncoder()
	le.fit(column)
	encoded_column = le.transform(column)
	del le
	return encoded_column

# Basic regression (taken from http://scikit-learn.org/stable/modules/linear_model.html)
# Only uses the numeric variables (i.e. condition and shipping method)
reg = linear_model.LinearRegression()
reg.fit(np.array([input_train_txt[:,2],input_train_txt[:,6]]).T, target_train)
first_prediction = reg.predict()

