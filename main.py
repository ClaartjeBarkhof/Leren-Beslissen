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

def validation_split(data, ratio):
	max_rounds = 1
	kf = KFold(n_splits=10, shuffle = True, random_state=42)
	kf.get_n_splits(data)
	error_list = []
	bias_list = []
	counter = 0
	for train_index, test_index in kf.split(data):
		train_data, test_data = data.iloc[train_index], data.iloc[test_index]
		train_data = train_data.reset_index(drop = True)
		test_data = test_data.reset_index(drop = True)
		train_X, test_X, train_y, test_y, train_split_X, test_split_X, train_split_y, test_split_y = preprocessing.preprocessing_main(train_data, test_data)	
#		prediction = learning_algorithms.splitted_learning(train_X, test_X, train_y, test_y, train_split_X, train_split_y, test_split_X, test_split_y, ratio)
		prediction2 = learning_algorithms.lgbmRidge(train_X, train_y, test_X, test_y)
		(error, bias) = analyse.calc_error(prediction)
		print("Error:", error)
		print("Bias:", bias)
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

def plot():
	print(main(1.00, clean_data=True))
	a = np.arange(0, 1.01, 0.1)
	e = []
	for i in range(11):
		print(i)
		e.append(main(a[i], clean_data = True))
	print(a)
	print(e)
	plt.scatter(a, e)
	plt.show()

def main(ratio, clean_data=False):
	if clean_data:
		clean_data = clean.clean_main()
	else:
		fileObject = open('../clean_matrix.pickle','rb')
		clean_data = pickle.load(fileObject)
	#training_set, training_target, validation_set, validation_target = validation_split(clean_data, 0.8)
	(error_list,bias_list) = validation_split(clean_data, ratio)
	#best_dim = analyse.analyse_main(training_set, training_target, validation_set, validation_target)
	#prediction = learning_algorithms.lgbmRidge(train_X, train_Y, test_X, test_Y)
	#(error, bias) = analyse.calc_error(prediction)
	error = sum(error_list)/float(len(error_list))
	bias = sum(bias_list)/float(len(bias_list))
	print("Bias: ")
	print(bias)
	print("Error: ")
	print(error)
	return(error)

plot()
#main(clean_data=True)
