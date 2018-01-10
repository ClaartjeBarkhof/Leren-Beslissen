import pandas as pd

def open_tsv(filepath):
	data = pd.read_table(filepath)
	return data#.iloc[0:10,:]

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

def main():
	data = open_tsv("../train.tsv")
	data = replace_NAN(data)
	data = split_catagories(data)
	data.to_csv("../cleaned.csv", sep='\t')

main()

