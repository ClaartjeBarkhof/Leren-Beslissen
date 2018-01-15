import pandas as pd
import nltk
import numpy as np
# nltk.download('tokenizer')
# nltk.download('corpus')
# nltk.download('stem')
# nltk.download('stopwords')
import time

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer


ps = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))

# First five encoders for categories. Sixth for brands
label_encoder_list = [LabelEncoder() for i in range(6)]
oh_encoder_list = [OneHotEncoder() for i in range(6)]

standard_scaler = preprocessing.StandardScaler()

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


# def compute_cleaned_size(data):
# 	rows, columns = (data.shape[0], data.shape[1])
# 	uc = len(np.unique(data['category_name']))
# 	ub = len(np.unique(data['brand_name']))
# 	columns = columns + uc + ub - 2

def binary_encoding(column, l_encoder, oh_encoder):
 	# ERROR als met meer dan 100 testen
 	# print("column")
 	# print(column)
 	# print(column)
 	# l_encoder = l_encoder.fit(column)
 	# column_int = l_encoder.transform(column)
 	# print(column_int)
 	# column_int = column_int.reshape(-1, 1)
 	lb = LabelBinarizer(sparse_output=True)
 	column_bin = lb.fit_transform(column).toarray()

 	# oh_encoder = oh_encoder.fit(column_int)
 	# column_bin = oh_encoder.transform(column_int).toarray()

# 	column_int = l_encoder.fit_transform(column.ravel()).reshape(*column.shape)
# 	column_int = column_int.reshape(-1, 1)
# 	column_bin = oh_encoder.fit_transform(column_int).toarray()



 	return(pd.DataFrame(column_bin))
#	return np.array(column_bin)

def bin_cleaning_data(data):
	new_data = pd.concat([data['train_id'], data['item_condition_id'], data['shipping']], axis=1)
#	matrix = np.vstack(np.array(data['train_id']), 
	for i in range(5):
		if 'category_'+str(i) in data.columns:
			new_data = pd.concat([new_data, binary_encoding(data['category_'+str(i)], label_encoder_list[i], oh_encoder_list[i])], axis=1)
#			new_data = data.drop('category_'+str(i), 1)		
	new_data = pd.concat([new_data, binary_encoding(data['brand_name'], label_encoder_list[5], oh_encoder_list[5])], axis=1)
	new_data = pd.concat([new_data, data['price']], axis=1)
#	data = data.drop('brand_name', 1)
	return(new_data.as_matrix())


def scale(data):
	print('data shipping')
	print(data[:,2])
	# standard_scaler.fit(data[:,1])
	# data[:,1] = standard_scaler.transform(data[:,1])

	standard_scaler.fit(np.transpose(data[:,2]))
	data[:,2] = standard_scaler.transform(np.transpose(data[:,2]))

	print(data[:,2])
	return(data)
	

#	X_train_minmax = min_max_scaler.fit_transform(X_train)

def clean_main():
#	print("hallo")
	data = open_tsv("../train.tsv")
	print(data.shape)
	data = data.iloc[0:100]
	t_start = time.time()
	data = replace_NAN(data)

	print("----%s seconds ----" %(time.time()-t_start))
	t_1 = time.time()

	data = split_catagories(data)

	print("----%s seconds ----" %(time.time()-t_1))
	t_2 = time.time()

	data = bin_cleaning_data(data)
	print("----%s seconds ----" %(time.time()-t_2))

#	data = scale(data)

	return data
#	print(data[0:10])
#	data.to_csv('../cleaned_binary.csv', sep=',')


clean_main()
	