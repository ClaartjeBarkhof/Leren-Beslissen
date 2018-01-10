import numpy as np
import pandas as pd

from sklearn import linear_model
from sklearn import metrics
from collections import Counter
from scipy.stats.stats import pearsonr

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

def validation_split(data, ratio):
    np.random.shuffle(data)
    training_size = round(len(data) * ratio)
    training_set = data[:training_size]
    validation_set = data[training_size:]
    return training_set, validation_set

# Testing the label encoder
print (input_train['category_name'].iloc[0:3])
print (label_encoder(input_train['category_name'].iloc[0:3]))
# Load the data
data = pd.read_table("train.tsv")
data_txt = data.as_matrix()

# Analyse most common categories
most_common_categories = Counter(data['category_name']).most_common(10)
for i in range(10):
    data_new = data.loc[data['category_name'] == most_common_categories[i][0]]
    data_new_txt = data_new.as_matrix()
    print(pearsonr(data_new_txt[:,2], data_new_txt[:,5]))

# Create training- and validation-set
training_set, validation_set = validation_split(data_new_txt, 0.7)
training_target = training_set[:,5]
validation_target = validation_set[:,5]
print("Data is loaded")

print(pearsonr(data_txt[:,2], data_txt[:,5]))

# Basic regression (taken from http://scikit-learn.org/stable/modules/linear_model.html)
reg = linear_model.LinearRegression()
reg.fit(np.array([training_set[:,2],training_set[:,6]]).T, training_target)
first_prediction = reg.predict(np.array([validation_set[:,2],validation_set[:,6]]).T)
error = metrics.mean_squared_error(first_prediction, validation_target)
print(error)