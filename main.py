import warnings
#warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
import clean
import learning_algorithms
import warnings
from sklearn.model_selection import train_test_split
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
import category_encoders as ce
from patsy import dmatrices
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import analyse
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
import preprocessing

import random

#		train_X, test_X, train_y, test_y, train_split_X, test_split_X, train_split_y, test_split_y = preprocessing.preprocessing_main(train_data, test_data)	
#		prediction = learning_algorithms.splitted_learning(train_X, test_X, train_y, test_y, train_split_X, train_split_y, test_split_X, test_split_y)


def validation_split(data):
	error_list = []
	bias_list = []
	X = data.drop(["price"],axis = 1)
	y = data["price"]

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
	train_data = pd.concat([X_train,y_train],axis = 1).reset_index(drop = True)
	test_data = pd.concat([X_test,y_test], axis = 1).reset_index(drop = True)
	rPerc = 0.2
	lPerc = 0.8
	train_X, test_X, train_y, test_y = preprocessing.preprocessing_main(train_data, test_data)
	for x in range(1):
		prediction = learning_algorithms.lgbmRidge(train_X, train_y, test_X, test_y, rPerc, lPerc)
		(error, bias) = analyse.calc_error(prediction)
		error_list.append(error)
		bias_list.append(bias)
		rPerc = rPerc - 0.04
		lPerc = lPerc + 0.04
#	lins = np.linspace(0.6,1.0,num =10)
	lins = np.linspace(0.6,1.0,num =1)

	plt.scatter(lins,error_list)
	plt.show()
	return error_list , bias_list

	# max_rounds = 5
	# kf = KFold(n_splits=3, shuffle = True, random_state=4)
	# kf.get_n_splits(data)
	# error_list = []
	# bias_list = []
	# counter = 0
	# for train_index, test_index in kf.split(data):
	# 	train_data, test_data = data.iloc[train_index], data.iloc[test_index]
	# 	train_data = train_data.reset_index(drop = True)
	# 	test_data = test_data.reset_index(drop = True)
	# 	train_X, test_X, train_y, test_y = preprocessing.preprocessing_main(train_data, test_data)
	# 	prediction = learning_algorithms.lgbmRidge(train_X, train_y, test_X, test_y, rPerc, lPerc)
	# 	(error, bias) = analyse.calc_error(prediction)
	# 	error_list.append(error)
	# 	bias_list.append(bias)
	# 	if counter == max_rounds:
	# 		break
	# 	counter += 1
	# 	print(counter)
	# return error_list , bias_list

# Expects a dataframe of one column:
# the predicted price
def write_submission(price_df, csv_name):
	id_df = pd.DataFrame(np.arange(price_df.shape[0]), columns=['test_id'])
	submission_df = pd.concat([id_df, price_df], axis=1)
	submission_df = submission_df.rename(columns = {'p':'price'})
	submission_df.to_csv(csv_name, sep=',', index=False)

# def plot():
# 	print(main(1.00, clean_data=True))
# 	a = np.arange(0, 1.01, 0.1)
# 	e = []
# 	for i in range(11):
# 		print(i)
# 		e.append(main(a[i], clean_data = True))
# 	print(a)
# 	print(e)
# 	plt.scatter(a, e)
# 	plt.show()

def main(clean_data=False):
	if clean_data:
		clean_data = clean.clean_main()
	else:
		fileObject = open('../clean_matrix.pickle','rb')
		clean_data = pickle.load(fileObject)
	#training_set, training_target, validation_set, validation_target = validation_split(clean_data, 0.8)
	(error_list,bias_list) = validation_split(clean_data)
	#best_dim = analyse.analyse_main(training_set, training_target, validation_set, validation_target)
	#prediction = learning_algorithms.lgbmRidge(train_X, train_Y, test_X, test_Y)
	#(error, bias) = analyse.calc_error(prediction)
	error = sum(error_list)/float(len(error_list))
	bias = sum(bias_list)/float(len(bias_list))
	print("Bias: ")
	print(bias)
	print("Error: ")
	print(error)
	return error

main(clean_data=True)