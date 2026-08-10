import pandas as pd

# Read dataset
data = pd.read_csv("enjoysport.csv")

print("Training Data:\n")
print(data)

# Attributes
concepts = data.iloc[:, :-1].values

# Target
target = data.iloc[:, -1].values

# Initialize hypothesis
hypothesis = concepts[0].copy()

print("\nInitial Hypothesis:")
print(hypothesis)

# FIND-S Algorithm
for i in range(len(concepts)):
    if target[i] == "Yes":
        for j in range(len(hypothesis)):
            if hypothesis[j] != concepts[i][j]:
                hypothesis[j] = "?"

print("\nFinal Hypothesis:")
print(hypothesis)