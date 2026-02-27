import pandas as pd

# Load dataset (make sure path is correct for your project)
df = pd.read_csv("data/creditcard.csv")

print("head of the dataset:", df.head())
print("Dataset shape:", df.shape)
print("Dataset columns:", df.columns)
print("Dataset info:", df.info())

print("Memory Usage:",df.memory_usage(deep=True))