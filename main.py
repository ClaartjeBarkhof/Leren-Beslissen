import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import clean

# Algorithm

# Cost function
# Expects a dataframe of two colums: 
# 		- col 1 to be the predicted price
#		- col 2 to be the actual price
def calc_error(dataframe):
	pred_price = dataframe['p']
	actual_price = dataframe['a']
	n = len(pred_price)
	error = np.sqrt((1 / n) * np.sum((np.log(pred_price + 1) - np.log(actual_price + 1)) ** 2))
	return error

# Expects a dataframe of one column:
# the predicted price
def write_submission(price_df, csv_name):
	id_df = pd.DataFrame(np.arange(price_df.shape[0]), columns=['test_id'])
	submission_df = pd.concat([id_df, price_df], axis=1)
	submission_df = submission_df.rename(columns = {'p':'price'})
	submission_df.to_csv(csv_name, sep=',', index=False)

def main():
	# TEST DATAFRAME
	d = {'p': [1, 2, 3, 4]}
	df = pd.DataFrame(data=d)
	#data = clean.open_tsv('../train.tsv')
	#print(len(set(data['category_name'])))
	#print(data.head())
	#print('RSMLE =',calc_error(df))
	write_submission(df, '../submission.csv')

main()
