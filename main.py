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
import analyse

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

def main(clean_data=False):
	if clean_data:
		clean_data = clean.clean_main()
	else:
		fileObject = open('../clean_matrix.pickle','rb')
		clean_data = pickle.load(fileObject)
	training_set, training_target, validation_set, validation_target = validation_split(clean_data, 0.8)
	best_dim = analyse.analyse_main(training_set, training_target, validation_set, validation_target)
	prediction = learning_algorithms.ridge(training_set, training_target, validation_set, validation_target)
	print(analyse.calc_error(prediction))

main(clean_data=True)