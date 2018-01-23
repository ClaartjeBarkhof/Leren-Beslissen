import numpy as np
from time import time
from scipy.stats import randint as sp_randint
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import load_digits
import lightgbm as lgb

param_grid = {
        'learning_rate': [0.001,0.005,0.01,0.05,0.1],
        # 'application': 'regression_l1',
        'max_depth': [1,3,4,5,7],
        'num_leaves': [90,120,180,210,250]
        # 'max_bin' :8192,
        # 'metric': 'RMSE',
        # 'data_random_seed': 2,
        # 'nthread': [2,3,4,5,6]
    }

def param_opt_main(X, y):
	print("dit is X",X)
	print("              ")
	print("Dit is Y",y)
	# X = [[1,5],[5,5],[3,8],[3,2],[6,7]]
	makeint = lambda x: int(x)
	func = np.vectorize(makeint)
	y = func(y)
	mdl = lgb.LGBMClassifier()
	# grid = RandomizedSearchCV(mdl, param_distributions=param_grid)
	# grid = GridSearchCV(mdl, param_grid=param_grid)
	grid = GridSearchCV(mdl, param_grid=param_grid)
	grid.fit(X,y)
	print(grid.best_params_)
	print(grid.best_score_)
	print("AAAAAAAAAAAAAAAAAAAAa")
