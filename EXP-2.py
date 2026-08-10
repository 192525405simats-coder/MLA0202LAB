import pandas as pd
import numpy as np

# Load dataset
data = pd.read_csv("enjoysport.csv")

concepts = np.array(data.iloc[:, :-1])
target = np.array(data.iloc[:, -1])

# Initialize S and G
S = concepts[0].copy()
G = [["?" for _ in range(len(S))]]

print("Initial Specific Hypothesis (S):")
print(S)

print("\nInitial General Hypothesis (G):")
print(G)

for i, h in enumerate(concepts):
    if target[i] == "Yes":
        # Generalize S
        for x in range(len(S)):
            if h[x] != S[x]:
                S[x] = "?"

        # Remove hypotheses from G inconsistent with positive example
        G = [g for g in G if all(g[j] == "?" or g[j] == h[j] for j in range(len(h)))]

    else:
        # Specialize G
        new_G = []
        for g in G:
            for x in range(len(S)):
                if g[x] == "?":
                    if S[x] != h[x]:
                        new_h = g.copy()
                        new_h[x] = S[x]
                        new_G.append(new_h)

        G = new_G

print("\nFinal Specific Hypothesis (S):")
print(S)

print("\nFinal General Hypothesis (G):")
for g in G:
    print(g)
