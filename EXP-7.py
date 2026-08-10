from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(
    iris.data,
    iris.target,
    test_size=0.3,
    random_state=42
)

model = LogisticRegression(max_iter=200)

model.fit(X_train, y_train)

prediction = model.predict(X_test)

print("Predicted Values:")
print(prediction)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, prediction))

print("\nAccuracy:")
print(accuracy_score(y_test, prediction))