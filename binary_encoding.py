from sklearn import preprocessing
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
#from sklearn.preprocessing import CategoricalEncoder

train_data = pd.read_table("../cleaned.csv")


def binary_encoding(column):
 	column_int = LabelEncoder().fit_transform(column.ravel()).reshape(*column.shape)
 	column_int = column_int.reshape(-1, 1)
 	column_bin = OneHotEncoder().fit_transform(column_int).toarray()
# 	print(column_bin)
# 	encoder = CategoricalEncoder(encoding='onehot-dense')
# 	binary = encoder.fit_transform(column)
# 	print(binary)
 	return(pd.DataFrame(column_bin))


def bin_cleaning_data(data):
	data = pd.concat([data, binary_encoding(data['category_name'])], axis=1)
	data = data.drop('category_name', 1)
	data = pd.concat([data, binary_encoding(data['brand_name'])], axis=1)
	data = data.drop('brand_name', 1)
	return(data)

#	print(new_data.loc[0])
#	new_data['category_name'] = binary_encoding(data['category_name'])new_data['category_name']
#	new_data['brand_name']

#binary = binary_encoding(['a', 'b', 'c', 'a', 'c', 'b'])

print(np.shape(train_data))
new_data = bin_cleaning_data(train_data.loc[0:10])

print(np.shape(train_data))
print(np.shape(new_data))
# data = pd.get_dummies(train_data.loc[0:10])
# print(data)

# binary = binary_encoding(train_data['brand_name'][0:10])
# print(train_data['brand_name'][0:10])
# print(binary)
#print(binary[0])