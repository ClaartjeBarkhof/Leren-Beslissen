import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
import nltk
import pickle as pickle
import operator

import numpy as np
import time
import category_encoders as ce
import copy

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import KFold
from sklearn.utils import shuffle

import analyse


MAX_FEATURES_ITEM_DESCRIPTION = 20000
MAX_FEATURES_ITEM_NAME = 20000



ps = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))

# First five encoders for categories. Sixth for brands
standard_scaler = preprocessing.StandardScaler()
oh_encoder_list = [ce.OneHotEncoder(handle_unknown="ignore") for i in range(6)]

bin_encoding_cols_dict = {}

tv = TfidfVectorizer(max_features=MAX_FEATURES_ITEM_DESCRIPTION, ngram_range=(1, 2), stop_words='english')
tv_name = TfidfVectorizer(max_features=MAX_FEATURES_ITEM_NAME, ngram_range=(1, 2), stop_words='english')

sentiment_analyzer = SentimentIntensityAnalyzer()


def TFidf_description(data, train):
	try:
		data['item_description']
	except KeyError:
		return data

	if train == True:
		tf_idf = tv.fit_transform(data['item_description']).toarray()
	else:
		tf_idf = tv.transform(data['item_description']).toarray()

	tf_idf = analyse.PCA_dimred(tf_idf, 1)
	tf_idf = pd.DataFrame(tf_idf)
	data = pd.concat([data, tf_idf], axis = 1)
	return data

def TFidf_name(data, train):
	try:
		data['name']
	except KeyError:
		return data

	if train == True:
		tf_idf_name = tv_name.fit_transform(data['name']).toarray()
	else:
		tf_idf_name = tv_name.transform(data['name']).toarray()

	tf_idf_name = analyse.PCA_dimred(tf_idf_name, 1)
	tf_idf_name = pd.DataFrame(tf_idf_name)
	data = pd.concat([data, tf_idf_name], axis = 1)
#	data['name'] = tf_idf_name
	return data


def binary_encoding(column, oh_encoder, train, category_1):

	if train == True:
		oh_encoder = oh_encoder.fit(np.array(column))
		
	if train == True and column.isnull().sum == column.shape[0]:
		column_bin = pd.DataFrame(np.zeros([column.shape[0], 1]))
	elif train == False and column.isnull().sum() == column.shape[0]:
		column_bin = pd.DataFrame(np.zeros([column.shape[0], bin_encoding_cols_dict[column.name]]))
	else:
		column_bin = oh_encoder.transform(np.array(column))

	if train == True:
		cols = column_bin.shape[1]
		bin_encoding_cols_dict[column.name] = cols

	if category_1 == True:
		column_bin = column_bin.rename(columns = lambda x: x.replace('0_', 'category_'))

	return column_bin

def bin_cleaning_data(data, train):
	standard_categories = ['name', 'item_condition_id', 'shipping', 'item_description', 'price']
	new_data = pd.DataFrame()
	for cat in standard_categories:
		if cat in data.columns:
			new_data = pd.concat([new_data, data[cat]], axis=1)
	category_1 = False
	#new_data = pd.concat([data['item_condition_id'], data['shipping'], data['item_description'], data['price']], axis=1)
	for i in range(5):
		if i == 0:
			category_1 =True
		else:
			category_1 = False
		if 'category_'+str(i) in data.columns:
			new_data = pd.concat([new_data, binary_encoding(data['category_'+str(i)], oh_encoder_list[i], train, category_1)], axis=1)

	try:
		new_data = pd.concat([new_data, binary_encoding(data['brand_name'], oh_encoder_list[5], train, False)], axis=1)
	except KeyError:
		return new_data
	
	return new_data

def drop_missing_brandnames(data):
	try:
		data['brand_name']
	except KeyError:
		return data

	rows_before_dropping = data.shape[0]
	data = data[(data.brand_name != 'undefined')]
	rows_after_dropping = data.shape[0]
	dropped_rows = rows_before_dropping - rows_after_dropping
	data = data.reset_index(drop=True)
	return data

def tuple_to_string(brand):
	result = ""
	for elm in brand:
		result += " " + elm
	return result[1:]

def record_most_common_brandnames_per_cat(train_data, test_data):
	data = pd.concat([train_data, test_data])
	data = data.reset_index(drop=True)
	mc_brandnames_per_cat = {}
	unique_cats = list(set(data['category_name']))
	for cat in unique_cats:
		x = data['brand_name'].loc[(data['category_name'] == cat)]
		counts = x.value_counts().index.tolist()
		if (len(counts) > 1) and (counts[0] == 'undefined'):
			mc_brandnames_per_cat[cat] = counts[1]
		else:
			mc_brandnames_per_cat[cat] = counts[0]
	return mc_brandnames_per_cat

def fill_in_missing_most_common_brandnames_per_cat(data, mc_brandnames_per_cat):
	try:
		data['category_name']
		data['brand_name']
	except KeyError:
		return data 
	for index, row in data.iterrows():
			if data.loc[index].brand_name == 'undefined':
				data.at[index, 'brand_name'] = mc_brandnames_per_cat[row['category_name']]
	return data

def replace_undefined_brand(item_name, brand_name, unique_brands):
	if brand_name == 'undefined':
		tokens = item_name.split(" ")
		tokens.extend(list(zip(tokens, tokens[1:])))
		intersection = set(tokens) & set(unique_brands)
		if intersection:
			brand = intersection.pop()
			while intersection:
				new_brand = intersection.pop()
				if type(new_brand) == tuple:
					brand = new_brand
					return tuple_to_string(brand)
			return brand
		else:
			return "undefined"
	else:
		return brand_name

def find_brands(train_data, test_data):
	data = pd.concat([train_data, test_data])
	data = data.reset_index(drop=True)
	all_brands = data['brand_name']
	unique_brands_raw = list(set(all_brands) - {'undefined'})
	unique_brands = []
	for elm in unique_brands_raw:
		if " " in elm:
			tokens = elm.split(" ")
			unique_brands.append(tuple(tokens))
		else:
			unique_brands.append(elm)
	return unique_brands


def fill_in_brand(data, unique_brands):
	try:
		data['brand_name']
	except KeyError:
		return data

	data['brand_name'] = data.apply(lambda row: replace_undefined_brand(row['name'], row['brand_name'], unique_brands), axis=1)

	try:
		data['item_description']
	except KeyError:
		return data
	data['brand_name'] = data.apply(lambda row: replace_undefined_brand(row['item_description'], row['brand_name'], unique_brands), axis=1)

	return data

def fill_in_brand_test(data, unique_brands):
	try:
		data['brand_name']
		data['name']
	except KeyError:
		return data
	data['brand_name'] = data.apply(lambda row: replace_undefined_brand(row['name'], row['brand_name'], unique_brands), axis=1)

	try:
		data['item_description']
	except KeyError:
		return data
	data['brand_name'] = data.apply(lambda row: replace_undefined_brand(row['item_description'], row['brand_name'], unique_brands), axis=1)
	return data

def get_sentiment(data):
	try:
		data['item_description']
	except KeyError:
		return data
	data['sentiment'] = data.apply(lambda row: sentiment_analyzer.polarity_scores(row['item_description'])['compound'], axis=1)
	return data

def cluster_train(data):
	db = DBSCAN(min_samples = 2).fit(data)
	labels = db.labels_
	print(len(set(labels)))
	print(set(labels))

def split(clean_data, ratio):
	clean_data = shuffle(clean_data)
	clean_data = clean_data.drop(['train_id'], axis=1)
	length = clean_data.shape[0]
	train_data = clean_data.iloc[0:int(length*ratio), :]
	test_data = clean_data.iloc[int(length*ratio):, :]
	train_data = train_data.reset_index(drop=True)
	test_data = test_data.reset_index(drop=True)
	return train_data, test_data

def horizontal_split(train_data, test_data):
	train_list = []
	test_list = []
	for x in range(11):
		column_name = 'category_'+str(x)
		if column_name in train_data.columns and column_name in test_data.columns:
			train_dataframe = train_data.loc[train_data[column_name] == 1]
			test_dataframe = test_data.loc[test_data[column_name] == 1]
		else:
			train_dataframe = None
			test_dataframe = None
		train_list.append(train_dataframe)
		test_list.append(test_dataframe)
	return(train_list, test_list)

# input cleaned dataframes, outputs 2 matrices
def preprocessing_main(train_data, test_data):
	#train_data, test_data = split(clean_data, 0.7)
	train_data = train_data.drop(['train_id'], axis=1)
	test_data = test_data.drop(['train_id'], axis=1)

	mc_brandnames_per_cat = record_most_common_brandnames_per_cat(train_data, test_data)
	unique_brands = find_brands(train_data, test_data)
	
	# TRAIN
	# Missing brand_names
	train_data = fill_in_brand(train_data, unique_brands)
	train_data = fill_in_missing_most_common_brandnames_per_cat(train_data, mc_brandnames_per_cat)
	#train_data = drop_missing_brandnames(train_data)

	# Price = 0 droppen
	train_data = train_data[(train_data.price > 0)]
	train_data = train_data.reset_index(drop=True)

	# Vul de categorieën die je niet mee wil nemen in. 
	# drop_categories = ['train_id', 'item_description', 'brand_name']
	# train_data = train_data.drop(drop_categories, axis=1)
	# test_data = test_data.drop(drop_categories, axis=1)

	train_data = bin_cleaning_data(train_data, True)
#	train_data = TFidf_description(train_data, True)
	train_data = TFidf_name(train_data, True)

	#train_data = get_sentiment(train_data)


	# TEST
	# Missing brand_names
	test_data = fill_in_brand(test_data, unique_brands)
	test_data = fill_in_missing_most_common_brandnames_per_cat(test_data, mc_brandnames_per_cat)
	test_data = bin_cleaning_data(test_data, False)
#	test_data = TFidf_description(test_data, False)
	test_data = TFidf_name(test_data, False)


	#test_data = get_sentiment(test_data)

	# item_description moet altijd gedropt worden 
	train_data = train_data.drop(['item_description', 'name'], axis=1)
	test_data = test_data.drop(['item_description', 'name'], axis=1)

	train_horizontal_splitted, test_horizontal_splitted = horizontal_split(train_data, test_data)
	train_X_splitted, train_y_splitted, test_X_splitted, test_y_splitted  = [], [], [], []
	for train, test in zip(train_horizontal_splitted, test_horizontal_splitted):
		train_y_splitted.append(train['price'])
		train_X_splitted.append(train.drop(['price'], axis=1))
		test_y_splitted.append(test['price'])
		test_X_splitted.append(test.drop(['price'], axis=1))

	train_Y = train_data['price'].as_matrix()
	train_X = train_data.drop(['price'], axis=1).as_matrix()

	test_Y = test_data['price'].as_matrix()
	test_X = test_data.drop(['price'], axis=1).as_matrix()

#	return train_X, test_X, train_Y, test_Y
	return train_X, test_X, train_Y, test_Y, train_X_splitted, test_X_splitted, train_y_splitted, test_y_splitted