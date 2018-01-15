import numpy as np
import pandas as pd

from sklearn import neural_network
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge

# Train_data bestaat nog niet!!!!
def decisiontree_regression(training_set, training_target, validation_set, validation_target):
	prediction_model = DecisionTreeRegressor()
	prediction_model.fit(train_data, train_target)
	prediction = prediction_model.predict(val_data)
	return pd.Dataframe({'p':prediction, 'a':validation_target})

def linear_regression(training_set, training_target, validation_set, validation_target):
	reg = linear_model.LinearRegression()
	reg.fit(training_set, training_target)
	prediction = reg.predict(validation_set)
	return pd.DataFrame({'p':prediction, 'a':validation_target})

def ann_regression(training_set, training_target, validation_set, validation_target):
	prediction_model = neural_network.MLPRegressor()
	prediction_model.fit(training_set, training_target)
	prediction = prediction_model.predict(validation_set)
	return pd.DataFrame({'p':prediction, 'a':validation_target})

	# eventueel solver='sag', fit_intercept=True
def ridge(training_set, training_target, validation_set, validation_target):
	prediction_model = Ridge(alpha=.6, copy_X=True, fit_intercept=True, max_iter=100,
		normalize=False, random_state=101, solver='auto', tol=0.01)
	prediction_model.fit(training_set, training_target)
	prediction = prediction_model.predict(validation_set)
	return pd.DataFrame({'p':prediction, 'a':validation_target})
