import pandas as pd
import numpy as np
import math
from collections import Counter
from nltk.tokenize import word_tokenize
import re

input_train = pd.read_table("../train.tsv")
input_train_txt = input_train.as_matrix()
target_train = input_train['price']
print(target_train.shape)