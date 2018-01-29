import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
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

from sklearn import linear_model
from sklearn import metrics
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge

def validation_split(data, cats):
	max_rounds = 5
	kf = KFold(n_splits=3, shuffle = True, random_state=4)
	kf.get_n_splits(data)
	error_list = []
	bias_list = []
	counter = 0
	for train_index, test_index in kf.split(data):
		train_data, test_data = data.iloc[train_index], data.iloc[test_index]
		train_data = train_data.reset_index(drop = True)
		test_data = test_data.reset_index(drop = True)
		train_X, test_X, train_y, test_y = preprocessing.preprocessing_main(train_data, test_data, cats)
		prediction = learning_algorithms.ridge(train_X, train_y, test_X, test_y)
		prediction.loc[prediction['p'] < 0, 'p'] = 0
		(error, bias) = analyse.calc_error(prediction)
		error_list.append(error)
		bias_list.append(bias)
		if counter == max_rounds:
			break
		counter += 1
	return error_list , bias_list

# Expects a dataframe of one column:
# the predicted price
def write_submission(price_df, csv_name):
	id_df = pd.DataFrame(np.arange(price_df.shape[0]), columns=['test_id'])
	submission_df = pd.concat([id_df, price_df], axis=1)
	submission_df = submission_df.rename(columns = {'p':'price'})
	submission_df.to_csv(csv_name, sep=',', index=False)

def main(cats, clean_data=False, feature_selection=False):
	if clean_data:
		clean_data = clean.clean_main()
	else:
		fileObject = open('../clean_matrix.pickle','rb')
		clean_data = pickle.load(fileObject)


	#training_set, training_target, validation_set, validation_target = validation_split(clean_data, 0.8)
	(error_list,bias_list) = validation_split(clean_data, cats)
	#best_dim = analyse.analyse_main(training_set, training_target, validation_set, validation_target)
	#prediction = learning_algorithms.lgbmRidge(train_X, train_Y, test_X, test_Y)
	#(error, bias) = analyse.calc_error(prediction)
	error = sum(error_list)/float(len(error_list))
	bias = sum(bias_list)/float(len(bias_list))
	#print("Bias: ")
	#print(bias)
	#print("Error: ")
	#print(error)
	return error, bias


all_cats = [[], ['name'], ['item_condition_id'], ['name', 'item_condition_id'], ['category_name'], ['name', 'category_name'], ['item_condition_id', 'category_name'], ['name', 'item_condition_id', 'category_name'], ['brand_name'], ['name', 'brand_name'], ['item_condition_id', 'brand_name'], ['name', 'item_condition_id', 'brand_name'], ['category_name', 'brand_name'], ['name', 'category_name', 'brand_name'], ['item_condition_id', 'category_name', 'brand_name'], ['name', 'item_condition_id', 'category_name', 'brand_name'], ['item_description'], ['name', 'item_description'], ['item_condition_id', 'item_description'], ['name', 'item_condition_id', 'item_description'], ['category_name', 'item_description'], ['name', 'category_name', 'item_description'], ['item_condition_id', 'category_name', 'item_description'], ['name', 'item_condition_id', 'category_name', 'item_description'], ['brand_name', 'item_description'], ['name', 'brand_name', 'item_description'], ['item_condition_id', 'brand_name', 'item_description'], ['name', 'item_condition_id', 'brand_name', 'item_description'], ['category_name', 'brand_name', 'item_description'], ['name', 'category_name', 'brand_name', 'item_description'], ['item_condition_id', 'category_name', 'brand_name', 'item_description'], ['name', 'item_condition_id', 'category_name', 'brand_name', 'item_description'], ['shipping'], ['name', 'shipping'], ['item_condition_id', 'shipping'], ['name', 'item_condition_id', 'shipping'], ['category_name', 'shipping'], ['name', 'category_name', 'shipping'], ['item_condition_id', 'category_name', 'shipping'], ['name', 'item_condition_id', 'category_name', 'shipping'], ['brand_name', 'shipping'], ['name', 'brand_name', 'shipping'], ['item_condition_id', 'brand_name', 'shipping'], ['name', 'item_condition_id', 'brand_name', 'shipping'], ['category_name', 'brand_name', 'shipping'], ['name', 'category_name', 'brand_name', 'shipping'], ['item_condition_id', 'category_name', 'brand_name', 'shipping'], ['name', 'item_condition_id', 'category_name', 'brand_name', 'shipping'], ['item_description', 'shipping'], ['name', 'item_description', 'shipping'], ['item_condition_id', 'item_description', 'shipping'], ['name', 'item_condition_id', 'item_description', 'shipping'], ['category_name', 'item_description', 'shipping'], ['name', 'category_name', 'item_description', 'shipping'], ['item_condition_id', 'category_name', 'item_description', 'shipping'], ['name', 'item_condition_id', 'category_name', 'item_description', 'shipping'], ['brand_name', 'item_description', 'shipping'], ['name', 'brand_name', 'item_description', 'shipping'], ['item_condition_id', 'brand_name', 'item_description', 'shipping'], ['name', 'item_condition_id', 'brand_name', 'item_description', 'shipping'], ['category_name', 'brand_name', 'item_description', 'shipping'], ['name', 'category_name', 'brand_name', 'item_description', 'shipping'], ['item_condition_id', 'category_name', 'brand_name', 'item_description', 'shipping'], ['name', 'item_condition_id', 'category_name', 'brand_name', 'item_description', 'shipping']]

results = open("results.txt","w") 

for cats in all_cats:
	print(cats)
	error, bias = main(cats, clean_data=True)
	result = '"'+str(cats)+'"'+","+'"'+str(error)+'"'+","+'"'+str(bias) +'"'+ "\n"
	results.write(result)

results.close()