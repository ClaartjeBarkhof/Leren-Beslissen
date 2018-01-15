import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
import clean
import learning_algorithms
import warnings
from sklearn.model_selection import train_test_split
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
import category_encoders as ce

# Cost function
# Expects a dataframe of two colums:
# 		- col 1 to be the predicted price
#		- col 2 to be the actual price
def calc_error(dataframe):
	pred_price = dataframe['p']
	actual_price = dataframe['a']
	print('\n',pred_price, '\n')
	print('\n',actual_price, '\n')
	n = len(pred_price)
	verschil_vec = (pred_price - actual_price)
	mean_verschil = (1 / n) * np.sum(np.absolute(verschil_vec))
	variance = (1 / n) * np.sum((verschil_vec - mean_verschil) ** 2)
	print("Gemiddelde afwijking in prijs:", mean_verschil)
	print("Variantie:", variance)
	error = np.sqrt((1 / n) * np.sum((np.log(pred_price + 1) - np.log(actual_price + 1)) ** 2))
	return error

def validation_split(data, ratio):
	t_x, v_x, t_y, v_y = train_test_split( data[:,:-1], data[:,-1], test_size=1-ratio, random_state=42)
	return t_x, t_y, v_x, v_y

# Expects a dataframe of one column:
# the predicted price
def write_submission(price_df, csv_name):
	id_df = pd.DataFrame(np.arange(price_df.shape[0]), columns=['test_id'])
	submission_df = pd.concat([id_df, price_df], axis=1)
	submission_df = submission_df.rename(columns = {'p':'price'})
	submission_df.to_csv(csv_name, sep=',', index=False)

def main():
	clean_data = clean.clean_main()
#	print()
	training_set, training_target, validation_set, validation_target = validation_split(clean_data, 0.8)

	print(prediction)
	prediction = learning_algorithms.ann_regression(training_set, training_target, validation_set, validation_target)
	return calc_error(prediction)
