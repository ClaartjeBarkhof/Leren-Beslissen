import numpy as np
import pandas as pd
import lightgbm as lgb
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

params = {
        'learning_rate': 0.1,
        'application': 'fair',
        'max_depth': 3,
        'num_leaves': 130,
        'verbosity': -1,
        'max_bin' :8192,
        'metric': 'RMSE',
        'data_random_seed': 2,
        'bagging_fraction': 1,
        'nthread': 4
    }

params2 = {
        'learning_rate': 0.15,
        'application': 'regression_l2',
        'max_depth': 3,
        'num_leaves': 130,
        'verbosity': -1,
        'max_bin' :8192,
        'metric': 'RMSE',
        'data_random_seed': 2,
        'bagging_fraction': 1,
        'nthread': 4
    }

def lgbm(training_set, training_target, validation_set, validation_target):
	d_train = lgb.Dataset(training_set, label=training_target)
	d_valid = lgb.Dataset(validation_set, label=validation_target)
	watchlist = [d_train, d_valid]
	model = lgb.train(params, train_set=d_train, num_boost_round=7500, valid_sets=watchlist, \
		early_stopping_rounds=1000, verbose_eval=1000)
	predsL = model.predict(validation_set)

	d_train2 = lgb.Dataset(training_set, label=training_target)
	d_valid2 = lgb.Dataset(validation_set, label=validation_target)
	watchlist2 = [d_train2, d_valid2]
	model2 = lgb.train(params2, train_set=d_train2, num_boost_round=7500, valid_sets=watchlist2, \
		early_stopping_rounds=1000, verbose_eval=1000)
	predsL2 = model2.predict(validation_set)
	return pd.DataFrame({'p':0.5*predsL+0.5*predsL2, 'a':validation_target})

def lgbmRidge(training_set, training_target, validation_set, validation_target):
	prediction_model = Ridge()
	prediction_model.fit(training_set, training_target)
	prediction = prediction_model.predict(validation_set)
	d_train = lgb.Dataset(training_set, label=training_target)
	d_valid = lgb.Dataset(validation_set, label=validation_target)
	watchlist = [d_train, d_valid]
	model = lgb.train(params, train_set=d_train, num_boost_round=7500, valid_sets=watchlist, \
		early_stopping_rounds=1000, verbose_eval=1000)
	predsL = model.predict(validation_set)
	return pd.DataFrame({'p':(0.5*predsL+0.5*prediction),'a':validation_target})
