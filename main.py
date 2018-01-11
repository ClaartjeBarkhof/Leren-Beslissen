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

def main():
	# TEST DATAFRAME
	#d = {'p': [1, 2], 'a': [3, 4]}
	#df = pd.DataFrame(data=d)
	data = clean.open_tsv('../train.tsv')
	print(data.head())
	#print('RSMLE =',calc_error(df))

main()
