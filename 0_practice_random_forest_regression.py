import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2].values

"""handling missing values"""
# from sklearn.impute import SimpleImputer
# imputer = SimpleImputer(missing_values=np.nan,strategy="median")
# X[:,:] = imputer.fit_transform(X[:,:])

"""handling categorical data"""
# from sklearn.preprocessing import LabelEncoder,OneHotEncoder
# label_encoder = LabelEncoder()
# X = label_encoder.fit_transform(X)
# X = np.reshape(X,(-1,1))
# from sklearn.compose import ColumnTransformer
# col_transformer = ColumnTransformer([('encoder',OneHotEncoder(),[0])],
#                                     remainder = "passthrough")
# X = (col_transformer.fit_transform(X)).toarray()
"""Splitting the data"""
# from sklearn.model_selection import train_test_split
# train_x,train_y,test_x,test_y = train_test_split(X,Y,test_size = 0.3)
"""Standardizing the data"""
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# sc_Y = StandardScaler()
# X =sc_X.fit_transform(X)
# Y = Y.reshape(-1,1)
# Y = sc_Y.fit_transform(Y)
# Y = Y.reshape(-1)
"""modeling the data"""
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300,criterion="mse")
regressor.fit(X,Y)
y_pred = regressor.predict([[6.5]])

"""plotting the data"""
X_grid = np.arange(min(X),max(X),0.01)
X_grid = np.reshape(X_grid,(len(X_grid),1))
plt.scatter(X,Y,color = "red")
plt.plot(X_grid,regressor.predict(X_grid),color = "blue")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.title("Truth or Bluff (Random Forest)")
plt.show()


