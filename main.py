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

def validation_split(data, features):
	error_list = []
	bias_list = []
	X = data.drop(["price"],axis = 1)
	y = data["price"]
	train, test, y_train, y_test = train_test_split(X, y, test_size=0.15)
	train_X, test_X = preprocessing.preprocessing_main2(train, test, features)
	for x in range(3):
		prediction = learning_algorithms.external_learning(train_X, np.log1p(y_train), test_X)
		(error, bias) = analyse.calc_error(prediction, y_test)
		error_list.append(error)
		bias_list.append(bias)
	lins = np.linspace(0.6,1.0,num =3)
	plt.scatter(lins,error_list)
	plt.show()
	return error_list, bias_list

def write_submission(price_df, csv_name):
	id_df = pd.DataFrame(np.arange(price_df.shape[0]), columns=['test_id'])
	submission_df = pd.concat([id_df, price_df], axis=1)
	submission_df = submission_df.rename(columns = {'p':'price'})
	submission_df.to_csv(csv_name, sep=',', index=False)

def main(features=[], clean_data=True):
	if clean_data:
		clean_data = clean.clean_main()
	else:
		fileObject = open('../clean_matrix.pickle','rb')
		clean_data = pickle.load(fileObject)
	(error_list,bias_list) = validation_split(clean_data, features)
	error = sum(error_list)/float(len(error_list))
	bias = sum(bias_list)/float(len(bias_list))
	print("Bias: ")
	print(bias)
	print("Error: ")
	print(error)
	return error

# Probleem #1: als je alleen "brand_fill" gebruikt gaat het fout omdat hstack raar omgaat met alleen een Dataframe als input
# Probleem #2: als je "shipping" toevoegd gaat LGBM klagen; hij verwacht een float maar krijgt een int. Dit komt doordat als je een sparsematrix aan een lijst toevoegd het type verandert naar dataframe
main([])