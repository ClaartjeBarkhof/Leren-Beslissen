import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
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
import ctypes  # An included library with Python install.   
from sklearn import linear_model
from sklearn import metrics
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge

def validation_split(data, cats):
	error_list = []
	bias_list = []
	X = data.drop(["price"],axis = 1)
	y = data["price"]

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=10)
	train_data = pd.concat([X_train,y_train],axis = 1).reset_index(drop = True)
	test_data = pd.concat([X_test,y_test], axis = 1).reset_index(drop = True)
	rPerc = 0.2
	lPerc = 0.8
	train_X, test_X, train_y, test_y = preprocessing.preprocessing_main(train_data, test_data, cats)
	for x in range(1):
		prediction = learning_algorithms.lgbmRidge(train_X, train_y, test_X, test_y, rPerc, lPerc)
		(error, bias) = analyse.calc_error(prediction)
		error_list.append(error)
		bias_list.append(bias)
	return error_list , bias_list

def write_submission(price_df, csv_name):
	id_df = pd.DataFrame(np.arange(price_df.shape[0]), columns=['test_id'])
	submission_df = pd.concat([id_df, price_df], axis=1)
	submission_df = submission_df.rename(columns = {'p':'price'})
	submission_df.to_csv(csv_name, sep=',', index=False)

def main(cats, clean_data=False):
	if clean_data:
		clean_data = clean.clean_main(cats)
	else:
		fileObject = open('../clean_matrix.pickle','rb')
		clean_data = pickle.load(fileObject)


	#training_set, training_target, validation_set, validation_target = validation_split(clean_data, 0.8)
	(error_list,bias_list) = validation_split(clean_data, cats)

	error = sum(error_list)/float(len(error_list))
	bias = sum(bias_list)/float(len(bias_list))
	#print("Bias: ")
	#print(bias)
	#print("Error: ")
	#print(error)
	return error, bias

all_cats = [[], ['name'], ['item_condition_id'], ['name', 'item_condition_id'], ['category_name'], ['name', 'category_name'], ['item_condition_id', 'category_name'], ['name', 'item_condition_id', 'category_name'], ['brand_name'], ['name', 'brand_name'], ['item_condition_id', 'brand_name'], ['name', 'item_condition_id', 'brand_name'], ['category_name', 'brand_name'], ['name', 'category_name', 'brand_name'], ['item_condition_id', 'category_name', 'brand_name'], ['name', 'item_condition_id', 'category_name', 'brand_name'], ['item_description'], ['name', 'item_description'], ['item_condition_id', 'item_description'], ['name', 'item_condition_id', 'item_description'], ['category_name', 'item_description'], ['name', 'category_name', 'item_description'], ['item_condition_id', 'category_name', 'item_description'], ['name', 'item_condition_id', 'category_name', 'item_description'], ['brand_name', 'item_description'], ['name', 'brand_name', 'item_description'], ['item_condition_id', 'brand_name', 'item_description'], ['name', 'item_condition_id', 'brand_name', 'item_description'], ['category_name', 'brand_name', 'item_description'], ['name', 'category_name', 'brand_name', 'item_description'], ['item_condition_id', 'category_name', 'brand_name', 'item_description'], ['name', 'item_condition_id', 'category_name', 'brand_name', 'item_description'], ['shipping'], ['name', 'shipping'], ['item_condition_id', 'shipping'], ['name', 'item_condition_id', 'shipping'], ['category_name', 'shipping'], ['name', 'category_name', 'shipping'], ['item_condition_id', 'category_name', 'shipping'], ['name', 'item_condition_id', 'category_name', 'shipping'], ['brand_name', 'shipping'], ['name', 'brand_name', 'shipping'], ['item_condition_id', 'brand_name', 'shipping'], ['name', 'item_condition_id', 'brand_name', 'shipping'], ['category_name', 'brand_name', 'shipping'], ['name', 'category_name', 'brand_name', 'shipping'], ['item_condition_id', 'category_name', 'brand_name', 'shipping'], ['name', 'item_condition_id', 'category_name', 'brand_name', 'shipping'], ['item_description', 'shipping'], ['name', 'item_description', 'shipping'], ['item_condition_id', 'item_description', 'shipping'], ['name', 'item_condition_id', 'item_description', 'shipping'], ['category_name', 'item_description', 'shipping'], ['name', 'category_name', 'item_description', 'shipping'], ['item_condition_id', 'category_name', 'item_description', 'shipping'], ['name', 'item_condition_id', 'category_name', 'item_description', 'shipping'], ['brand_name', 'item_description', 'shipping'], ['name', 'brand_name', 'item_description', 'shipping'], ['item_condition_id', 'brand_name', 'item_description', 'shipping'], ['name', 'item_condition_id', 'brand_name', 'item_description', 'shipping'], ['category_name', 'brand_name', 'item_description', 'shipping'], ['name', 'category_name', 'brand_name', 'item_description', 'shipping'], ['item_condition_id', 'category_name', 'brand_name', 'item_description', 'shipping']]
five_best = [[], ['name', 'item_description', 'shipping'], ['name']]
results = open("five_best.txt","w") 

for cats in five_best:
	print(cats)
	error, bias = main(list(cats), clean_data=True)
	result = '"'+str(cats)+'"'+","+'"'+str(error)+'"'+","+'"'+str(bias) +'"'+ "\n"
	results.write(result)

ctypes.windll.user32.MessageBoxW(0, "DONE!", "results", 1)

results.close()
