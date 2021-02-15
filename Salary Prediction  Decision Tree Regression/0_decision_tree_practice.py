import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#getting the data 
dataset = pd.read_csv("Position_Salaries.csv") 
X  = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2].values

"""handling missing data"""
# from sklearn.impute import SimpleImputer
# imputer = SimpleImputer(missing_values=np.nan,strategy = "median")
# X = imputer.fit_transform(X)
"""Handling categorical data"""
# from sklearn.preprocessing import LabelEncoder,OneHotEncoder
# from sklearn.compose import ColumnTransformer


# col_transformer = ColumnTransformer([("encoder",OneHotEncoder(),[0])],
#                                       remainder = "passthrough")
# X = (col_transformer.fit_transform(X)).toarray()
"""splitting the data"""
# from sklearn.model_selection import train_test_split
# train_x,train_y,test_x,test_y = train_test_split(X,Y,test_size = 0.3)
"""standardizing the data"""
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# sc_Y = StandardScaler()
# X = sc_X.fit_transform(X)
# Y = sc_X.fit_transform(Y)
"""Decision tree"""
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(criterion="mse")
regressor.fit(X,Y)
y_pred = regressor.predict([[6.5]])
# from sklearn import tree
# tree.plot_tree(regressor)

X_grid = np.arange(min(X),max(X),step = 0.2)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,Y,color = "red")
plt.plot(X_grid,regressor.predict(X_grid),color = "blue")






