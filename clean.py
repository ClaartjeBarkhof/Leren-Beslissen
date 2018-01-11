import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer

ps = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))

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

def main():
	data = open_tsv("../train.tsv")
	#data = replace_NAN(data)
	#data = split_catagories(data)
	#data.to_csv("../cleaned.csv", sep='\t')

main()

