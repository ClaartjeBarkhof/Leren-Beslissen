import warnings
import gc
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

from scipy.sparse import csr_matrix, hstack

import analyse


MAX_FEATURES_ITEM_DESCRIPTION = 10000
MAX_FEATURES_ITEM_NAME = 10000
NUM_BRANDS = 4000
NUM_CATEGORIES = 1000
NAME_MIN_DF = 10


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
	if train == True:
		tf_idf = tv.fit_transform(data).toarray()
	else:
		tf_idf = tv.transform(data).toarray()
	return tf_idf

def TFidf_name(data, train):
	if train == True:
		tf_idf_name = tv_name.fit_transform(data).toarray()
	else:
		tf_idf_name = tv_name.transform(data).toarray()
	return tf_idf_name



def new_binary_encoding(column, oh_encoder):
	oh_encoder = oh_encoder.fit(np.array(column))
	column_bin = oh_encoder.transform(column)
	return column_bin
	

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
	standard_categories = ['name', 'item_condition_id', 'shipping', 'item_description', 'price', 'description_len']
	new_data = pd.DataFrame()
	for cat in standard_categories:
		if cat in data.columns:
			new_data = pd.concat([new_data, data[cat]], axis=1)
	print('BAD HERE??')
	print(new_data.head())
	category_1 = False
	#new_data = pd.concat([data['item_condition_id'], data['shipping'], data['item_description'], data['price']], axis=1)
	for i in range(5):
		if i == 0:
			category_1 =True
		else:
			category_1 = False
		if 'category_'+str(i) in data.columns:
			new_data = pd.concat([new_data, binary_encoding(data['category_'+str(i)], oh_encoder_list[i], train, category_1)], axis=1)

	print('not sow bad now')
	print(new_data.head())

	try:
		new_data = pd.concat([new_data, binary_encoding(data['brand_name'], oh_encoder_list[5], train, False)], axis=1)
	except KeyError:
		pass
	return new_data

def drop_missing_brandnames(data):
	data = data[(data != 'undefined')]
	data = data.reset_index(drop=True)
	return data

def tuple_to_string(brand):
	result = ""
	for elm in brand:
		result += " " + elm
	return result[1:]

def record_most_common_brandnames_per_cat(train_data, test_data):
	try:
		train_data['category_name']
		train_data['brand_name']
	except KeyError:
		return ['']

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

def find_brands(data):
	data = data.reset_index(drop=True)
	all_brands = data
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
	result = data.apply(lambda row: replace_undefined_brand(row['name'], row['brand_name'], unique_brands), axis=1)
	result = data.apply(lambda row: replace_undefined_brand(row['item_description'], row['brand_name'], unique_brands), axis=1)
	return result

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
	result = data.apply(lambda x: sentiment_analyzer.polarity_scores(x)['compound'])
	return result

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


def cutting(dataset):
    pop_brand = dataset['brand_name'].value_counts().loc[lambda x: x.index != 'undefined'].index[:NUM_BRANDS]
    dataset.loc[~dataset['brand_name'].isin(pop_brand), 'brand_name'] = 'undefined'
    pop_category = dataset['category_name'].value_counts().loc[lambda x: x.index != 'undefined'].index[:NUM_BRANDS]
    dataset.loc[~dataset['category_name'].isin(pop_category), 'category_name'] = 'undefined'


def description_len(data):
	#data['tokenized_description'] = data['item_description'].apply(lambda x: tokenize(x))
	return data.apply(lambda x: x.count(' '))

def get_name_bin(name_column):
	cv = CountVectorizer(max_features=20000, stop_words="english") # \b[A-z_][A-z_]+\b mogelijke regex om woorden met cijfers te skippen
	result = cv.fit_transform(name_column)
	return result

def preprocessing_main2(train_data, test_data, features):
	start_time = time.time()
	nrow_train = train_data.shape[0]
	merge = pd.concat([train_data, test_data])

	submission = test_data['train_id']
	del train_data
	del test_data

	gc.collect()

	final_features = []
	# print('[{}] Finished count vectorize `name`'.format(time.time() - start_time))

	X_category0 = new_binary_encoding(merge['category_0'].as_matrix(), oh_encoder_list[0])
	X_category1 = new_binary_encoding(merge['category_1'].as_matrix(), oh_encoder_list[1])
	X_category2 = new_binary_encoding(merge['category_2'].as_matrix(), oh_encoder_list[0])
	X_category3 = new_binary_encoding(merge['category_3'].as_matrix(), oh_encoder_list[1])
	X_category4 = new_binary_encoding(merge['category_4'].as_matrix(), oh_encoder_list[0])

	if "cat_1" in features:
		X_category = X_category0
		final_features.append(X_category)
	if "cat_2" in features:
		X_category = hstack((X_category0, X_category1))
		final_features.append(X_category)
	if "cat_3" in features:
		X_category = hstack((X_category0, X_category1, X_category2))
		final_features.append(X_category)
	if "cat_4" in features:
		X_category = hstack((X_category0, X_category1, X_category2, X_category3))
		final_features.append(X_category)
	if "cat_all" in features:
		X_category = hstack((X_category0, X_category1, X_category2, X_category3, X_category4))
		final_features.append(X_category)

	if "descr_tfidf" in features:
		X_description = TFidf_description(merge['item_description'], True)
		final_features.append(X_description)
	if "descr_sentiment" in features:
		X_sentiment = np.matrix(get_sentiment(merge['item_description'])).T
		final_features.append(X_sentiment)
	if "descr_len" in features:
		X_descr_len = np.matrix(description_len(merge['item_description'])).T
		final_features.append(X_descr_len)

	if "name_bin" in features:
		X_name_bin = get_name_bin(merge['name'])
		final_features.append(X_name_bin)
	if "name_tfidf" in features:
		X_name_tfidf = TFidf_name(merge['name'], True)
		final_features.append(X_name_tfidf)

	if "brand_fill" in features or "brand_both" in features:
		unique_brands = find_brands(merge['brand_name'])
		X_brand_fill = fill_in_brand(merge[['name','item_description','brand_name']], unique_brands)
		X_brand = new_binary_encoding(X_brand_fill.as_matrix(), oh_encoder_list[5])
		final_features.append(X_brand)
	if "brand_nothing" in features:
		X_brand = new_binary_encoding(merge['brand_name'].as_matrix(), oh_encoder_list[5])
		final_features.append(X_brand)

	if "shipping" in features and "item_condition" in features:
		X_dummies = csr_matrix(pd.get_dummies(merge[['item_condition_id', 'shipping']],
                                          sparse=True).values)
		final_features.append(X_dummies)
	elif "shipping" in features:
		X_shipping = csr_matrix(pd.get_dummies(merge['shipping'],
                                          sparse=True).values)
		final_features.append(X_shipping)
	elif "item_condition" in features:
		X_condition = csr_matrix(pd.get_dummies(merge[['item_condition_id']],
                                          sparse=True).values)
		final_features.append(X_condition)
	for elm in final_features:
		print(elm.shape)
		print(type(elm))
	sparse_merge = hstack((final_features)).astype(float).tocsr()
	X = sparse_merge[:nrow_train]
	X_test = sparse_merge[nrow_train:]

	return X, X_test

"""
def preprocessing_main3(train_data, test_data, features):
	start_time = time.time()
	nrow_train = train_data.shape[0]
	merge = pd.concat([train_data, test_data])

	submission = test_data['train_id']
	del train_data
	del test_data

	gc.collect()

	# print('[{}] Finished count vectorize `name`'.format(time.time() - start_time))
	X_name = TFidf_name(merge['name'], True)

	X_category0 = new_binary_encoding(merge['category_0'].as_matrix(), oh_encoder_list[0])
	X_category1 = new_binary_encoding(merge['category_1'].as_matrix(), oh_encoder_list[1])
	X_category2 = new_binary_encoding(merge['category_2'].as_matrix(), oh_encoder_list[0])
	X_category3 = new_binary_encoding(merge['category_3'].as_matrix(), oh_encoder_list[1])
	X_category4 = new_binary_encoding(merge['category_4'].as_matrix(), oh_encoder_list[0])

	X_category = hstack((X_category0, X_category1, X_category2, X_category3, X_category4))
	# print('[{}] Finished TFIDF vectorize `item_description`'.format(time.time() - start_time))
	X_description = TFidf_description(merge['item_description'], True)

	# print('[{}] Finished label binarize `brand_name`'.format(time.time() - start_time))
	X_brand = new_binary_encoding(merge['brand_name'].as_matrix(), oh_encoder_list[5])

	X_dummies = csr_matrix(pd.get_dummies(merge[['item_condition_id', 'shipping']],
                                          sparse=True).values)
	print('[{}] Finished to get dummies on `item_condition_id` and `shipping`'.format(time.time() - start_time))

	list1 = [X_dummies, X_description, X_brand, X_category, X_name]
	for elem in list1:
		print(type(elem))

	sparse_merge = hstack((X_dummies, X_description, X_brand, X_category, X_name)).tocsr()
	print('[{}] Finished to create sparse merge'.format(time.time() - start_time))

	

	X = sparse_merge[:nrow_train]
	X_test = sparse_merge[nrow_train:]


	print('MY HEAD')
	print(X[0:10])

	print("TRAINING SHAPE")
	print(X.shape)
	print("Test SHAPE")
	print(X_test.shape)
	return X, X_test
"""
