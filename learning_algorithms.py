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

params1 = {
        'learning_rate': 0.1,
        'application': 'regression_l1',
		# 'boosting': 'rf',
        'max_depth': 4,
        'num_leaves': 180,
        'verbosity': -1,
        'max_bin' :8192,
        'metric': 'RMSE',
        'data_random_seed': 2,
        # 'bagging_fraction': 1,
        'nthread': 4,
        'min_data': 1,
        'min_data_in_bin': 1


    }

params2 = {
        'learning_rate': 0.85,
        'application': 'regression_l1',
        'max_depth': 3,
        'num_leaves': 100,
        'verbosity': -1,
        'metric': 'RMSE',
        'data_random_seed': 2,
        # 'bagging_fraction': 1,
        'nthread': 4,
        'min_data': 1,
        'min_data_in_bin': 1

    }

def lgbm(training_set, training_target, validation_set, validation_target):
	d_train = lgb.Dataset(training_set, label=training_target)
	d_valid = lgb.Dataset(validation_set, label=validation_target)
	watchlist = [d_train, d_valid]
	model = lgb.train(params1, train_set=d_train, num_boost_round=7500, valid_sets=watchlist, \
		early_stopping_rounds=1000, verbose_eval=1000)
	predsL = model.predict(validation_set)
	return pd.DataFrame({'p':predsL, 'a':validation_target})

def lgbmRidge(training_set, training_target, validation_set, validation_target):
	R1prediction_model = Ridge()
	R1prediction_model.fit(training_set, training_target)
	R1prediction = R1prediction_model.predict(validation_set)

	R2prediction_model = Ridge(alpha=.6, copy_X=True, fit_intercept=True, max_iter=100,
		normalize=False, random_state=101, solver='auto', tol=0.01)
	R2prediction_model.fit(training_set, training_target)
	R2prediction = R2prediction_model.predict(validation_set)

	lgbm1_train = lgb.Dataset(training_set, label=training_target)
	lgbm1_valid = lgb.Dataset(validation_set, label=validation_target)
	watchlist = [lgbm1_train, lgbm1_valid]
	lgbm1_model = lgb.train(params1, train_set=lgbm1_train, num_boost_round=7500, valid_sets=watchlist, early_stopping_rounds=1000, verbose_eval=1000)
	lgbm1_pred = lgbm1_model.predict(validation_set)

	lgbm2_train = lgb.Dataset(training_set, label=training_target)
	lgbm2_valid = lgb.Dataset(validation_set, label=validation_target)
	watchlist2 = [lgbm2_train, lgbm2_valid]
	lgbm2_model = lgb.train(params2, train_set=lgbm2_train, num_boost_round=7500, valid_sets=watchlist2, \
		early_stopping_rounds=1000, verbose_eval=1000)
	lgbm2_pred = lgbm2_model.predict(validation_set)
	return pd.DataFrame({'p':(0.5*lgbm1_pred+0.25*R1prediction+0.15*lgbm2_pred+0.1*R2prediction),'a':validation_target})
	

def splitted_learning(train_X, test_X, train_y, test_y, train_X_split, train_y_split, test_X_split, test_y_split, ratio):
	prediction = lgbmRidge(train_X, train_y, test_X, test_y)
	p = pd.Series()
	for trainx, trainy, testx, testy in zip(train_X_split, train_y_split, test_X_split, test_y_split):
		split_prediction = lgbmRidge(trainx, trainy, testx, testy)
		p = pd.concat([p, split_prediction['p']], axis=0)

	p = p.sort_index(axis = 0)

	prediction['p'] = prediction['p'].apply(lambda x: x*ratio)
	p = p.apply(lambda x: x*(1-ratio))
	prediction['p'] = prediction['p'].add(p)


#
	return(prediction)