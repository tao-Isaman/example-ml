# summarize_data.py

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
import os

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

# Save Pairplot as Image
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

sns.pairplot(df, hue="target_name")
plt.suptitle("Pairplot of Iris Features by Class", y=1.02)

# Save the plot
output_path = os.path.join(output_dir, "iris_pairplot.png")
plt.savefig(output_path)
print(f"\nPairplot saved to {output_path}")

# Show plot (optional, remove if running on server without GUI)
plt.show()
