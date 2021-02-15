import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2].values
"""handling missing values"""
# from sklearn.impute import SimpleImputer
# X = SimpleImputer(missing_values = np.nan,strategy="median").fit_transform(X)

"""handling categorical data"""
# from sklearn.preprocessing import OneHotEncoder,LabelEncoder
# X = LabelEncoder().fit_transform(X)
# from sklearn.compose import ColumnTransformer
# col_transformer = ColumnTransformer([("encoding",OneHotEncoder(),[0])],remainder="passthrough")
# X = np.array(col_transformer.fit_transform(X),dtype=int)
"""Splitting the data"""
# from sklearn.model_selection import train_test_split
# train_x,train_y,test_x,test_y = train_test_split(X,Y,test_size = 0.3)
"""Standardizing the data"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(Y.reshape((-1,1)))

"""Modeling our data using svr"""
from sklearn.svm import SVR
regressor = SVR(kernel="rbf")
regressor.fit(X,Y.reshape(-1))
y_pred = regressor.predict([[6.5]])
y_pred = sc_Y.inverse_transform(y_pred)
#visualizing the plot
plt.scatter(X,Y,color="red")
plt.plot(X,regressor.predict(X),color="blue")
plt.title("SVR Truth or bluff")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()
#smooth graph
X_grid = np.arange(min(X),max(X),step = 0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,Y,color = "red")
plt.plot(X_grid,regressor.predict(X_grid),color = "blue")
plt.title("Smooth SVR Truth or bluff")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()


