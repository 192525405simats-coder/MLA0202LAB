import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv("salary_data.csv")

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

linear = LinearRegression()
linear.fit(X, y)

poly = PolynomialFeatures(degree=2)

X_poly = poly.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)

plt.scatter(X, y, color="blue")

plt.plot(X, linear.predict(X), color="red", label="Linear")

plt.plot(
    X,
    poly_model.predict(X_poly),
    color="green",
    label="Polynomial"
)

plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Linear vs Polynomial Regression")
plt.legend()
plt.show()