import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn import neural_network
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split


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


def external_learning(X, y, X_test):
	start_time = time.time()
	model = Ridge(solver="sag", fit_intercept=True, random_state=205, alpha=3)
	model.fit(X, y)
	print('[{}] Finished to train ridge sag'.format(time.time() - start_time))
	predsR = model.predict(X=X_test)
	print('[{}] Finished to predict ridge sag'.format(time.time() - start_time))

	model = Ridge(solver="lsqr", fit_intercept=True, random_state=145, alpha = 3)
	model.fit(X, y)
	print('[{}] Finished to train ridge lsqrt'.format(time.time() - start_time))
	predsR2 = model.predict(X=X_test)
	print('[{}] Finished to predict ridge lsqrt'.format(time.time() - start_time))

	train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size = 0.1, random_state = 144) 
	d_train = lgb.Dataset(train_X, label=train_y)
	d_valid = lgb.Dataset(valid_X, label=valid_y)
	watchlist = [d_train, d_valid]
    
	params = {
		'learning_rate': 0.76,
		'application': 'regression',
		'max_depth': 3,
		'num_leaves': 99,
		'verbosity': -1,
		'metric': 'RMSE',
		'nthread': 4
	}

	params2 = {
		'learning_rate': 0.85,
		'application': 'regression',
		'max_depth': 3,
		'num_leaves': 110,
		'verbosity': -1,
		'metric': 'RMSE',
		'nthread': 4
	}

	model = lgb.train(params, train_set=d_train, num_boost_round=7500, valid_sets=watchlist, \
	early_stopping_rounds=500, verbose_eval=500) 
	predsL = model.predict(X_test)
    
	print('[{}] Finished to predict lgb 1'.format(time.time() - start_time))
    
	train_X2, valid_X2, train_y2, valid_y2 = train_test_split(X, y, test_size = 0.1, random_state = 101) 
	d_train2 = lgb.Dataset(train_X2, label=train_y2)
	d_valid2 = lgb.Dataset(valid_X2, label=valid_y2)
	watchlist2 = [d_train2, d_valid2]

	model = lgb.train(params2, train_set=d_train2, num_boost_round=3000, valid_sets=watchlist2, \
	early_stopping_rounds=50, verbose_eval=500) 
	predsL2 = model.predict(X_test)

	print('[{}] Finished to predict lgb 2'.format(time.time() - start_time))

	preds = predsR2*0.15 + predsR*0.15 + predsL*0.5 + predsL2*0.2
	preds = np.expm1(preds)
	return preds


def lgbmRidge(training_set, training_target, test_set, rPerc,lPerc):
	trainX, validationX, trainY, validationY = train_test_split(training_set, training_target, test_size=0.1, random_state=10)

	print("TRAINY")
	print(training_target[0:10])


	R1prediction_model = Ridge()
	R1prediction_model.fit(trainX, trainY)
	R1prediction = R1prediction_model.predict(test_set)

	# R2prediction_model = Ridge(alpha=.6, copy_X=True, fit_intercept=True, max_iter=100,
	# 	normalize=False, random_state=101, solver='auto', tol=0.01)
	# R2prediction_model.fit(training_set, training_target)
	# R2prediction = R2prediction_model.predict(validation_set)

	lgbm1_train = lgb.Dataset(trainX, label=trainY)
	lgbm1_valid = lgb.Dataset(validationX, label=validationY)
	watchlist = [lgbm1_train, lgbm1_valid]
	lgbm1_model = lgb.train(params1, train_set=lgbm1_train, num_boost_round=7500, valid_sets=watchlist, early_stopping_rounds=1000, verbose_eval=1000)
	lgbm1_pred = lgbm1_model.predict(test_set)
	# lgbm2_train = lgb.Dataset(training_set, label=training_target)
	# lgbm2_valid = lgb.Dataset(validation_set, label=validation_target)
	# watchlist2 = [lgbm2_train, lgbm2_valid]
	# lgbm2_model = lgb.train(params2, train_set=lgbm2_train, num_boost_round=7500, valid_sets=watchlist2, \
	# 	early_stopping_rounds=1000, verbose_eval=1000)
	# lgbm2_pred = lgbm2_model.predict(validation_set)

#	prediction = np.expm1(lPerc*lgbm1_pred+rPerc*R1prediction)
#	validation_target = np.expm1(validation_target)
	prediction = (lPerc*lgbm1_pred+rPerc*R1prediction)

#	prediction = np.expm1(prediction)

	return(pd.DataFrame(prediction))
#	return pd.DataFrame({'p':(prediction),'a':test_target})



def splitted_learning(rPerc, lPerc, train_X, test_X, train_y, train_X_split, train_y_split, test_X_split):
	prediction = lgbmRidge(train_X, train_y, test_X, rPerc, lPerc)
	p = pd.Series()
	for trainx, trainy, testx in zip(train_X_split, train_y_split, test_X_split):
		split_prediction = lgbmRidge(trainx, trainy, testx, rPerc, lPerc)
		p = pd.concat([p, split_prediction], axis=0)

	p = p.sort_index(axis = 0)

	prediction = prediction.apply(lambda x: x*0.7)
	p = p.apply(lambda x: x*(0.3))
	prediction = prediction.add(p)
	return(prediction)
