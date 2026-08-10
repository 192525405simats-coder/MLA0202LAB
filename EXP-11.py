import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

data = {
    'Age': [25, 35, 45, 23, 52, 40, 29, 48, 31, 55],
    'Income': [25000, 50000, 75000, 22000, 90000, 65000, 35000, 85000, 45000, 100000],
    'LoanAmount': [10000, 20000, 15000, 12000, 25000, 18000, 14000, 22000, 16000, 30000],
    'CreditHistory': [1, 1, 1, 0, 1, 1, 0, 1, 1, 1],
    'CreditScore': [
        'Poor', 'Good', 'Excellent', 'Poor', 'Excellent',
        'Good', 'Poor', 'Excellent', 'Good', 'Excellent'
    ]
}

df = pd.DataFrame(data)

encoder = LabelEncoder()
df['CreditScore'] = encoder.fit_transform(df['CreditScore'])

X = df[['Age', 'Income', 'LoanAmount', 'CreditHistory']]
y = df['CreditScore']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Actual Values:")
print(encoder.inverse_transform(y_test))

print("\nPredicted Values:")
print(encoder.inverse_transform(y_pred))

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=range(len(encoder.classes_)),
    target_names=encoder.classes_,
    zero_division=0
))

print("Shaik Rafiqhuddin - 192525129")