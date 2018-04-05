# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import comet_ml in the top of your file
from comet_ml import Experiment

experiment  = Experiment(api_key="Dz2W3DAahv0OvSAERUfhA5b7I")

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, [0,1,2]].values
y = dataset.iloc[:, 3].values

# GeT to know your data
dataset.describe()
dataset.info()

# Check if missing values
dataset.isnull().values.any()

# Check number of NaNs
dataset.isnull().sum().sum()

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)