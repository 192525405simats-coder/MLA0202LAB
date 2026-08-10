import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

data = pd.read_csv("play_tennis.csv")

encoder = LabelEncoder()

for column in data.columns:
    data[column] = encoder.fit_transform(data[column])

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

model = DecisionTreeClassifier(criterion="entropy")
model.fit(X, y)

plt.figure(figsize=(10, 6))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=["No", "Yes"],
    filled=True
)
plt.show()