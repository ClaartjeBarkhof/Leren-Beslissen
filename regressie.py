
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import Ridge
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dropout, Dense, concatenate, GRU, Embedding, Flatten, Activation
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K

input_train = pd.read_table("../train.tsv")
input_train_txt = input_train.as_matrix()
target_train = input_train['price']
print(target_train.shape)

def error_function(labels, predicted):
    # Y and Y_red have already been in log scale.
    assert Y.shape == Y_pred.shape
    return np.sqrt(np.mean(np.square(predicted - labels)))

# Returns encoded discrete values of a single column
def label_encoder(column):
	le = labelEncoder()
	le.fit(column)
	return le.transform(column)