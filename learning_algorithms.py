import numpy as np
import pandas as pd

from sklearn import neural_network
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor

def decisiontree_regression(training_set, training_target, validation_set, validation_target):
	prediction_model = DecisionTreeRegressor()
	prediction_model.fit(train_data, train_target)
	prediction = prediction_model.predict(val_data)
    return pd.Dataframe({'p':prediction, 'a':validation_target})

def linear_regression(training_set, training_target, validation_set, validation_target):
    reg = linear_model.LinearRegression()
    reg.fit(training_set, training_target)
    prediction = reg.predict(validation_set)
    return pd.Dataframe({'p':prediction, 'a':validation_target})

def ann_regression(training_set, training_target, validation_set, validation_target):
	prediction_model = neural_network.MLPRegressor()
	prediction_model.fit(train_data, train_target)
	prediction = prediction_model.predict(val_data)
    return pd.Dataframe({'p':prediction, 'a':validation_target})