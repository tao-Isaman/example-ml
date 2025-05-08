# summarize_data.py

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Convert to DataFrame
df = pd.DataFrame(X, columns=iris.feature_names)
df["target"] = y
df["target_name"] = df["target"].map(dict(enumerate(iris.target_names)))

# Summary
print("==== Basic Information ====")
print(f"Shape: {df.shape}")
print("\nFeature Names:", iris.feature_names)
print("Target Names:", iris.target_names)

print("\n==== First 5 Rows ====")
print(df.head())

print("\n==== Class Distribution ====")
print(df["target_name"].value_counts())

print("\n==== Feature Statistics ====")
print(df.describe())

print("\n==== Correlation Matrix ====")
print(df.corr(numeric_only=True))

# OPTIONAL: Plotting (for visual understanding)
sns.pairplot(df, hue="target_name")
plt.suptitle("Pairplot of Iris Features by Class", y=1.02)
plt.show()
