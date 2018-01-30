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
import matplotlib.pyplot as plt
import seaborn as sns


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
#from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

#import analyse

INSTANCES = 1000


def open_tsv(filepath):
	data = pd.read_table(filepath)
	return data #.iloc[0:10,:]

def replace_NAN(data):
	data['category_name'] = data['category_name'].fillna('undefined').astype(str)
	data['brand_name'] = data['brand_name'].fillna('undefined').astype(str)
	data['item_description'] = data['item_description'].fillna('undefined')
	return data

def split_categories(data):
	column_split = lambda x: pd.Series([i for i in (x.split('/'))])
	splitted = data['category_name'].apply(column_split)
	num_categories = splitted.shape[1]
	data = pd.concat([data, splitted], axis=1)

	# while num_categories < 5:
	# 	data[num_categories] = np.nan
	# 	num_categories += 1

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

def scale(data):
	# standard_scaler.fit(data[:,1])
	# data[:,1] = standard_scaler.transform(data[:,1])

	standard_scaler.fit(np.transpose(data[:,2]))
	data[:,2] = standard_scaler.transform(np.transpose(data[:,2]))

	return(data)


def clean_main():
	t_start = time.time()
	data = open_tsv("../train.tsv")
	t_start = time.time()
	data = replace_NAN(data)
	#data = add_description_len(data)
	#data = split_categories(data)

	'''
	print("Undefined brand_name", len(data[(data.brand_name == 'undefined')]))
	missing_brandname_data_cat = data[(data.brand_name == 'undefined')].category_name
	counts = missing_brandname_data_cat.value_counts()
	length = counts.index.shape[0]
	list2 = [i for i in range(length)]
	sns.barplot(counts.values, list2)
	plt.show()	
	'''
	
	print()
	print("ClEANING TIME:")
	print("---- %s seconds ----" %(time.time()-t_start))
	# Save cleaned data matrix in file
	#fileName = '../clean_matrix.pickle'
	#fileObject = open(fileName,'wb')
	#pickle.dump(data, fileObject)
	#fileObject.close()
	#data = scale(data)
	return data

clean_main()
	