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

def validation_split(data, ratio):
    np.random.shuffle(data)
    training_size = round(len(data) * ratio)
    training_set = data[:training_size]
    validation_set = data[training_size:]
    return training_set, validation_set

def regression(training_set, training_target, validation_set, validation_target):
    reg = linear_model.LinearRegression()
    reg.fit(training_set, training_target)
    prediction = reg.predict(validation_set)
    return pd.Dataframe({'p':prediction, 'a':validation_target})

def main():
	training_set, training_target, validation_set, validation_target = clean.open_tsv('../train.tsv')
	prediction = regression(training_set, training_target, validation_set, validation_target)
	return clac_error(prediction)
	
main()
