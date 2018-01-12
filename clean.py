import pandas as pd
import nltk
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


ps = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))

def open_tsv(filepath):
	data = pd.read_table(filepath, nrows=10)
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

def binary_encoding(column):
 	column_int = LabelEncoder().fit_transform(column.ravel()).reshape(*column.shape)
 	column_int = column_int.reshape(-1, 1)
 	column_bin = OneHotEncoder().fit_transform(column_int).toarray()
 	return(pd.DataFrame(column_bin))
#	return np.array(column_bin)

def bin_cleaning_data(data):
#	matrix = np.array(data['train_id'])

	for i in range(5):
		if 'category_'+str(i) in data.columns:
			data = pd.concat([data, binary_encoding(data['category_'+str(i)])], axis=1)
			data = data.drop('category_'+str(i), 1)		
	data = pd.concat([data, binary_encoding(data['brand_name'])], axis=1)
	data = data.drop('brand_name', 1)
	return(data)


def clean_main():
	data = open_tsv("../train.tsv")
	data = data.iloc[0:100]
	t_start = time.time()
	data = replace_NAN(data)
	data = split_catagories(data)
	data = bin_cleaning_data(data)
	print(data)
	#data.to_csv('../cleaned_binary.csv', sep=',')
	#print("----%s seconds ----" %(time.time()-t_start))
#	print(data.head())

clean_main()
	#data.to_csv("../cleaned.csv", sep='\t')
