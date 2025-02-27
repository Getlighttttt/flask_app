import torch
import pandas as pd
import matplotlib.pyplot as plt

import MyKMeans as km


# Load dataset from Hugging Face
# df = pd.read_csv("hf://datasets/varun-d/demo-data/test-data.csv")
# df = pd.read_json("hf://datasets/lancewilhelm/cs6804_final_v2/data/e2c44e43-43e5-44bb-82f7-055a14ce1537.json", lines=True)
df = pd.read_csv("hf://datasets/sebastian-hofstaetter/tripclick-training/improved_tripclick_train_triple-ids.tsv", sep="\t")
# df = pd.read_csv("test_dataset.csv")
# df = pd.read_csv("hf://datasets/chungimungi/Colors/colors.csv")

# Select only numeric columns (float & int) and drop missing values
df_numeric = df.select_dtypes(include=['float64', 'int64']).dropna()

# Convert to PyTorch tensor
device = "cuda"
X = torch.tensor(df_numeric.values, dtype=torch.float32).to(device)


# Request user input
user_input = input("Enter a number: ")

# # Check if the input is a number
k = int(user_input) if user_input.isdigit() else None

while (k != None and k > X.shape[0]):
    print("k must be lower than row count")
    user_input = input("Enter a number: ")
    k = int(user_input) if user_input.isdigit() else None

user_input = input("Enter a Elbow method tolerance: ")

if user_input.isdigit():
    opt, wcss, g, l = km.elbow_method(X, k, device=device, tol=int(user_input)) 
else:
    opt, wcss,g,l = km.elbow_method(X, k, device=device)

print("Optimal number fo clusters: ", opt, wcss)

# df_numeric["Cluster"] = clusters.cpu().numpy()
# df_numeric.to_csv("clustered.csv", index=False)
# pd.DataFrame(centroids.cpu().numpy(), columns=df_numeric.columns[:-1]).to_csv("centroids.csv", index=False)

# print("Clustering complete! Results saved to clustered.csv and centroids.csv.")