import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
import clean
import learning_algorithms
import warnings
from sklearn.model_selection import train_test_split
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
import category_encoders as ce
from patsy import dmatrices
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Cost function
# Expects a dataframe of two colums:
# 		- col 1 to be the predicted price
#		- col 2 to be the actual price
def calc_error(dataframe):
	pred_price = dataframe['p']
	actual_price = dataframe['a']
	n = len(pred_price)
	verschil_vec = (pred_price - actual_price)
	mean_verschil = (1 / n) * np.sum(np.absolute(verschil_vec))
	variance = (1 / n) * np.sum((verschil_vec - mean_verschil) ** 2)
	#print("Gemiddelde afwijking in prijs:", mean_verschil)
	#print("Variantie:", variance)
	error = np.sqrt((1 / n) * np.sum((np.log(pred_price + 1) - np.log(actual_price + 1)) ** 2))
	return error

def PCA_dimred(matrix, dim):
	pca = PCA(n_components=dim)
	pca.fit(matrix)
	reduced_matrix = pca.transform(matrix)
	#print(pca.explained_variance_ratio_)
	return reduced_matrix

def validation_split(data, ratio):
	t_x, v_x, t_y, v_y = train_test_split( data[:,:-1], data[:,-1], test_size=1-ratio, random_state=40)
	return t_x, t_y, v_x, v_y

def calc_VIF(X, y):
	vif = pd.DataFrame()
	vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
	vif["features"] = X.columns
	print(vif.round(1))

# Expects a dataframe of one column:
# the predicted price
def write_submission(price_df, csv_name):
	id_df = pd.DataFrame(np.arange(price_df.shape[0]), columns=['test_id'])
	submission_df = pd.concat([id_df, price_df], axis=1)
	submission_df = submission_df.rename(columns = {'p':'price'})
	submission_df.to_csv(csv_name, sep=',', index=False)

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
	return dim_list, error_list, best_dim


def main(clean_data=True):
	if clean_data:
		clean_data = clean.clean_main()
	else:
		fileObject = open('../clean_matrix.pickle','rb')
		clean_data = pickle.load(fileObject)
	training_set, training_target, validation_set, validation_target = validation_split(clean_data, 0.8)
#	dim_list, error_list, best_dim = plot_PCA_options(training_set, training_target, validation_set, validation_target)
#	print("BEST DIMENSION:", best_dim[0], "with an error of:", best_dim[1])
	prediction = learning_algorithms.ridge(training_set, training_target, validation_set, validation_target)

	print(calc_error(prediction))

main(clean_data=True)