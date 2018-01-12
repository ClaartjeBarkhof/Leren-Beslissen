import pandas as pd
import nltk
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

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

ps = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))

# First five encoders for categories. Sixth for brands
oh_encoder_list = [ce.OneHotEncoder(handle_unknown="ignore") for i in range(6)]

def open_tsv(filepath):
	data = pd.read_table(filepath)
	return data #.iloc[0:10,:]

def replace_NAN(data):
	data['category_name'] = data['category_name'].fillna('undefined').astype(str)
	data['brand_name'] = data['brand_name'].fillna('undefined').astype(str)
	data['item_description'] = data['item_description'].fillna('undefined')
	return data

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

def add_tokenize_cols(data):
	data['tokenized_description'] = data['item_description'].apply(lambda x: tokenize(x))
	data['description_len'] = data['tokenized_description'].apply(lambda x: len(x))
	return

def binary_encoding(column, oh_encoder):
 	oh_encoder = oh_encoder.fit(column)
 	column_bin = oh_encoder.transform(column)
 	return column_bin

def bin_cleaning_data(data):
	new_data = pd.concat([data['train_id'], data['item_condition_id'], data['shipping']], axis=1)
	for i in range(5):
		if 'category_'+str(i) in data.columns:
			new_data = pd.concat([new_data, binary_encoding(data['category_'+str(i)], oh_encoder_list[i])], axis=1)
	new_data = pd.concat([new_data, binary_encoding(data['brand_name'], oh_encoder_list[5])], axis=1)
	new_data = pd.concat([new_data, data['price']], axis=1)
	return new_data.as_matrix()

def clean_main():
#	print("hallo")
	data = open_tsv("../train.tsv")
	print(data.shape)
	data = data.iloc[0:1000]
	t_start = time.time()
	data = replace_NAN(data)

	print("----%s seconds ----" %(time.time()-t_start))
	t_1 = time.time()

	data = split_catagories(data)

	print("----%s seconds ----" %(time.time()-t_1))
	t_2 = time.time()

	data = bin_cleaning_data(data)
	print("----%s seconds ----" %(time.time()-t_2))

	print(data.shape)
	return data
#	print(data[0:10])
#	data.to_csv('../cleaned_binary.csv', sep=',')


#clean_main()
	