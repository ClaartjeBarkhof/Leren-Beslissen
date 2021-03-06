import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import learning_algorithms
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
#import main


# Cost function
# Expects a dataframe of two colums:
# 		- col 1 to be the predicted price
#		- col 2 to be the actual price
def calc_error(prediction, testy):
#	pred_price = dataframe['p']
#	prediction = prediction.where(prediction > 0, 0)
#	prediction = prediction.as_matrix()
#	testy = testy.as_matrix()
	# print(pred_price)
	prediction[prediction < 3] = 3
	prediction[prediction > 2000] = 2000
	bias = np.mean(prediction)-np.mean(testy)
	n = len(prediction)
	verschil_vec = (prediction - testy)
	mean_verschil = (1 / n) * np.sum(np.absolute(verschil_vec))
	variance = (1 / n) * np.sum((verschil_vec - mean_verschil) ** 2)
	#print("Gemiddelde afwijking in prijs:", mean_verschil)
	#print("Variantie:", variance)
	error = np.sqrt(np.mean(np.power(np.log1p(prediction)-np.log1p(testy), 2)))
	error_list = np.sqrt((np.power(np.log1p(prediction)-np.log1p(testy), 2)))
	return error, bias, error_list

def plot_PCA_options(training_set, training_target, validation_set, validation_target):
	dim_list = []
	error_list = []
	dim = 1
	best_dim = (0, 100)
	for i in range(25):
		dim_list.append(dim)
		reduced_train_X = PCA_dimred(training_set, dim)
		reduced_valid_X = PCA_dimred(validation_set, dim)
		prediction = learning_algorithms.ridge(reduced_train_X, training_target, reduced_valid_X, validation_target)
		error = calc_error(prediction)
		error_list.append(error)
		if error < best_dim[1]:
			best_dim = (dim, error)
		dim += 1
	plt.scatter(dim_list, error_list)
	plt.show()
	# VIFS
	#reduced_train_X = PCA_dimred(training_set, best_dim[0])
	#print("VIF of best reduction:", calc_VIF(reduced_train_X, training_target))
	return dim_list, error_list, best_dim

def PCA_dimred(matrix, dim):
	pca = PCA(n_components=dim)
	pca.fit(matrix)
	reduced_matrix = pca.transform(matrix)
	#print(pca.explained_variance_ratio_)
	return reduced_matrix

def calc_VIF(X, y):
	X = pd.DataFrame(X)
	vif = pd.DataFrame()
	vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
	vif["features"] = X.columns
	print(vif.round(1))

def analyse_main(training_set, training_target, validation_set, validation_target):
	dim_list, error_list, best_dim = plot_PCA_options(training_set, training_target, validation_set, validation_target)
	print("BEST DIMENSION:", best_dim[0], "with an error of:", best_dim[1])
	return best_dim
