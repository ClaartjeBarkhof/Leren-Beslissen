import pandas as pd
import numpy as np
import nltk
#nltk.download()
import pickle as pickle
import operator

import numpy as np
# nltk.download('tokenizer')
# nltk.download('corpus')
# nltk.download('stem')
# nltk.download('stopwords')
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
import analyse

MAX_FEATURES_ITEM_DESCRIPTION = 10000

INSTANCES = 1000000

ps = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))

# First five encoders for categories. Sixth for brands
standard_scaler = preprocessing.StandardScaler()
oh_encoder_list = [ce.OneHotEncoder(handle_unknown="ignore") for i in range(6)]


def open_tsv(filepath):
	data = pd.read_table(filepath, nrows=INSTANCES)
	return data #.iloc[0:10,:]

def replace_NAN(data):
	data['category_name'] = data['category_name'].fillna('undefined').astype(str)
	data['brand_name'] = data['brand_name'].fillna('undefined').astype(str)
	data['item_description'] = data['item_description'].fillna('undefined')
	return data

def drop_missing_brandnames(data):
	print('# rows before dropping missing brandnames', data.shape[0])
	data = data[(data.brand_name == 'undefined')]

	print('# rows after dropping missing brandnames', data.shape[0])
	data = data.reset_index()
	return data

'''
def record_most_common_brandnames(data):
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

def fill_in_missing_brandnames(data):
	mc_brandnames_per_cat = record_most_common_brandnames(data)
	count = 0
	print(len(data['brand_name'].loc[(data.brand_name == 'undefined')]))
	for index, row in data.iterrows():
		if row['brand_name'] == 'undefined':
			row['brand_name'] = mc_brandnames_per_cat[row['category_name']]
			#print(row)
			#print('--------')
			#print(mc_brandnames_per_cat[row['category_name']])
			if mc_brandnames_per_cat[row['category_name']] != 'undefined':
				count += 1
	print(len(data['brand_name'].loc[(data.brand_name == 'undefined')]))
	#print(count)
'''

def split_catagories(data):
	column_split = lambda x: pd.Series([i for i in (x.split('/'))])
	splitted = data['category_name'].apply(column_split)
	data = pd.concat([data, splitted], axis=1)
	data = data.rename(columns = {0:'category_0', 1:"category_1", 2:"category_2", 3:"category_3", 4:"category_4"})	
	return data

def tokenize(description):
    try:
        split = tokenizer.tokenize(description)
        filtered_sentence = []
        #filtered_sentence1 = [w.lower() for w in split if not w in stop_words]
        filtered_sentence = [ps.stem(w.lower()) for w in split if not w in stop_words]
        # filtered_sentence = set(filtered_sentence) In case you would want to remove the duplicates.
        return filtered_sentence
    except: 
        print("error with description:", description)
        return []

def add_description_len(data):
	#data['tokenized_description'] = data['item_description'].apply(lambda x: tokenize(x))
	data['description_len'] = data['item_description'].apply(lambda x: x.count(' '))
	return data

def TFidf(data):
	price = data['price']
	tv = TfidfVectorizer(max_features=MAX_FEATURES_ITEM_DESCRIPTION, ngram_range=(1, 2), stop_words='english')
	tf_idf = tv.fit_transform(data['item_description']).toarray()
	tf_idf = analyse.PCA_dimred(tf_idf, 1)

	tf_idf = pd.DataFrame(tf_idf)
	#vocab = tv.vocabulary_
	#sorted_vocab = sorted(vocab.items(), key=operator.itemgetter(0))
	#print(sorted_vocab)
	new_data = data.drop(['item_description', 'price'], axis=1)
	new_data = pd.concat([new_data, tf_idf, price], axis = 1)
	return(new_data)

def binary_encoding(column, oh_encoder):
	oh_encoder = oh_encoder.fit(np.array(column))
	column_bin = oh_encoder.transform(np.array(column))
	return column_bin

def bin_cleaning_data(data):
	new_data = pd.concat([data['train_id'], data['item_condition_id'], data['shipping'], data['item_description']], axis=1)
#	new_data = data.drop(['name', 'price', 'brand_name', 'category_0', 'category_1', 'category_2', 'category_3', 'category_4'])
	for i in range(5):
		if 'category_'+str(i) in data.columns:
			new_data = pd.concat([new_data, binary_encoding(data['category_'+str(i)], oh_encoder_list[i])], axis=1)


	new_data = pd.concat([new_data, binary_encoding(data['brand_name'], oh_encoder_list[5])], axis=1)
	new_data = pd.concat([new_data, data['price']], axis=1)


#	return new_data.as_matrix()
	return new_data

def scale(data):
	# standard_scaler.fit(data[:,1])
	# data[:,1] = standard_scaler.transform(data[:,1])

	standard_scaler.fit(np.transpose(data[:,2]))
	data[:,2] = standard_scaler.transform(np.transpose(data[:,2]))

	return(data)

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

def find_brands(all_brands):
	unique_brands_raw = list(set(all_brands) - {'undefined'})
	unique_brands = []
	for elm in unique_brands_raw:
		if " " in elm:
			tokens = elm.split(" ")
			unique_brands.append(tuple(tokens))
		else:
			unique_brands.append(elm)
	return unique_brands

def fill_in_brand(data):
	unique_brands = find_brands(data['brand_name'])
	data['brand_name'] = data.apply(lambda row: replace_undefined_brand(row['name'], row['brand_name'], unique_brands), axis=1)
	return data

def get_sentiment(data):
	sentiment_analyzer = SentimentIntensityAnalyzer()
	data['item_description'] = data.apply(lambda row: sentiment_analyzer.polarity_scores(row['item_description'])['compound'], axis=1)
	return data

def clean_main():
	t_start = time.time()
	data = open_tsv("../train.tsv")
	t_start = time.time()
	data = replace_NAN(data)

	data = data[(data.price > 0)]
	data = data.reset_index()

	data = fill_in_brand(data)
	data = drop_missing_brandnames(data)

	#blabla = fill_in_missing_brandnames(data)

	data = add_description_len(data)
	data = split_catagories(data)
	data = bin_cleaning_data(data)
	data = get_sentiment(data)
#	data = data.drop(['item_description'], axis=1)
#	data = TFidf(data)
	data = data.as_matrix()
	print("ClEANING TIME:")
	print("---- %s seconds ----" %(time.time()-t_start))
	# Save cleaned data matrix in file
	#fileName = '../clean_matrix.pickle'
	#fileObject = open(fileName,'wb')
	#pickle.dump(data, fileObject)
	#fileObject.close()
	#data = scale(data)
	return data

#clean_main()
	