import pandas as pd
import numpy as np
import nltk
import pickle as pickle
import operator

import numpy as np
import time
import category_encoders as ce

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

MAX_FEATURES_ITEM_DESCRIPTION = 10000

ps = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))

# First five encoders for categories. Sixth for brands
standard_scaler = preprocessing.StandardScaler()
oh_encoder_list = [ce.OneHotEncoder(handle_unknown="ignore") for i in range(6)]

tv = TfidfVectorizer(max_features=MAX_FEATURES_ITEM_DESCRIPTION, ngram_range=(1, 2), stop_words='english')
sentiment_analyzer = SentimentIntensityAnalyzer()

def TFidf(data, train):
	if train == True:
		tf_idf = tv.fit_transform(data['item_description']).toarray()
	else:
		tf_idf = tv.transform(data['item_description']).toarray()
	tf_idf = analyse.PCA_dimred(tf_idf, 1)
	tf_idf = pd.DataFrame(tf_idf)
	data = pd.concat([data, tf_idf], axis = 1)
	return(data)

def binary_encoding(column, oh_encoder, train):
	if train == True:
		oh_encoder = oh_encoder.fit(np.array(column))
	column_bin = oh_encoder.transform(np.array(column))
	return column_bin

def bin_cleaning_data(data, train):
	new_data = pd.concat([data['item_condition_id'], data['shipping'], data['item_description']], axis=1)
	for i in range(5):
		if 'category_'+str(i) in data.columns:
			new_data = pd.concat([new_data, binary_encoding(data['category_'+str(i)], oh_encoder_list[i], train)], axis=1)
	new_data = pd.concat([new_data, binary_encoding(data['brand_name'], oh_encoder_list[5], train)], axis=1)
	new_data = pd.concat([new_data, data['price']], axis=1)

#	return new_data.as_matrix()
	return new_data

def tuple_to_string(brand):
	result = ""
	for elm in brand:
		result += " " + elm
	return result[1:]

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

def find_brands_train(all_brands):
	unique_brands_raw = list(set(all_brands) - {'undefined'})
	unique_brands = []
	for elm in unique_brands_raw:
		if " " in elm:
			tokens = elm.split(" ")
			unique_brands.append(tuple(tokens))
		else:
			unique_brands.append(elm)
	return unique_brands

def fill_in_brand_train(data):
	unique_brands = find_brands_train(data['brand_name'])
	data['brand_name'] = data.apply(lambda row: replace_undefined_brand(row['name'], row['brand_name'], unique_brands), axis=1)
	data['brand_name'] = data.apply(lambda row: replace_undefined_brand(row['item_description'], row['brand_name'], unique_brands), axis=1)
	return data, unique_brands

def fill_in_brand_test(data, unique_brands):
	data['brand_name'] = data.apply(lambda row: replace_undefined_brand(row['name'], row['brand_name'], unique_brands), axis=1)
	data['brand_name'] = data.apply(lambda row: replace_undefined_brand(row['item_description'], row['brand_name'], unique_brands), axis=1)
	return data

def get_sentiment(data):
	print(data.iloc[2])
	data['item_description'] = data.apply(lambda row: sentiment_analyzer.polarity_scores(row['item_description'])['compound'], axis=1)
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

# input cleaned dataframe, outputs 4 matrices 
def preprocessing_main(clean_data):
	train_data, test_data = split(clean_data, 0.7)

	# functies op train fitten
	train_data, unique_brands = fill_in_brand_train(train_data)
	train_data = TFidf(train_data, True)
	train_data = bin_cleaning_data(train_data, True)
	#train_data = get_sentiment(train_data)

	# functies op test toepassen
	test_data = fill_in_brand_test(test_data, unique_brands)
	test_data = TFidf(test_data, False)
	test_data = bin_cleaning_data(test_data, False)
	#test_data = get_sentiment(test_data)

	train_Y = train_data['price']
	train_X = train_data.drop(['price', 'item_description'], axis=1)
	test_Y = test_data['price']
	test_X = test_data.drop(['price', 'item_description'], axis=1)

	return train_X.as_matrix(), train_Y.as_matrix(), test_X.as_matrix(), test_Y.as_matrix()
			


