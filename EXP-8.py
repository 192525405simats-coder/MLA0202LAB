import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv("salary_data.csv")

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

model = LinearRegression()

model.fit(X, y)

prediction = model.predict(X)

plt.scatter(X, y)
plt.plot(X, prediction, color="red")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Linear Regression")
plt.show()