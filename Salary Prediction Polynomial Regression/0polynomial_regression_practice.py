import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,-1].values
"""handling missing values"""
# from sklearn.impute import SimpleImputer
# imputer = SimpleImputer(missing_values=np.nan,strategy="median")
# X = imputer.fit_transform(X[:,1])
"""handling categorical data"""

# from sklearn.preprocessing import LabelEncoder,OneHotEncoder
# from sklearn.compose import ColumnTransformer
# X[:,0] = LabelEncoder().fit_transform(X[:,0])
# col_transformer = ColumnTransformer(["encoder",OneHotEncoder(),[0] ],remainder="passthrough")
# X[:,0] = col_transformer.fit_transform(X[:,0])
"""scaling data"""
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X[:,0] = sc_X.fit_transform(X[:,0]) 
"""splitting the data"""
# from sklearn.model_selection import train_test_split
# train_x,train_y,train_x,test_x = train_test_split(X,Y,test_size = 0.3 random_state = 42)
from sklearn.linear_model import LinearRegression
lin_regressor = LinearRegression()
lin_regressor.fit(X,Y)
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_regressor = LinearRegression()
poly_regressor.fit(X_poly,Y)

plt.scatter(X,Y,color="red")
plt.plot(X,lin_regressor.predict(X),color="blue")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.title("Truth or Bluff")
plt.show()

plt.scatter(X,Y,color="red")
plt.plot(X,poly_regressor.predict(poly_reg.fit_transform(X)),color = "blue")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.title("Truth or Bluff")
plt.show()
# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(start = min(X),stop = max(X),step = 0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,Y,color="red")
plt.plot(X_grid,poly_regressor.predict(poly_reg.fit_transform(X_grid)))
plt.xlabel("Position")
plt.ylabel("Salary")
plt.title("Truth or Bluff")
plt.show()

print(lin_regressor.predict([[6.5]]))
print(poly_regressor.predict(poly_reg.fit_transform([[6.5]])))




