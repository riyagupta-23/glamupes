import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import random

from load_data import load_product_data
from normalize_ingredients import normalize_ingredient_list
from nlp_similarity import compute_embeddings
from sklearn.metrics.pairwise import cosine_similarity

# Task 3: NLP Similarity Heatmap of 10 Random Products

# Load and preprocess
df = load_product_data("data/makeup-products-list.csv")
df["normalized_ingredients"] = df["ingredients"].apply(normalize_ingredient_list)
df["normalized_ingredients"] = df["normalized_ingredients"].apply(lambda x: list(dict.fromkeys(x)))

# Compute embeddings
embeddings = compute_embeddings(df["normalized_ingredients"].tolist())

# Select 10 random product indices
random.seed(42)
random_indices = random.sample(range(len(df)), 10)
selected_names = [df.iloc[i]["glamupe_name"][:25] for i in random_indices]
selected_embeddings = [embeddings[i] for i in random_indices]

# Compute similarity matrix
sim_matrix = cosine_similarity(selected_embeddings)

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(sim_matrix, xticklabels=selected_names, yticklabels=selected_names,
            cmap='Blues', annot=True, fmt=".2f", square=True, linewidths=0.5)

plt.title("NLP Similarity Heatmap of 10 Random Products")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

# Save figure
os.makedirs("data", exist_ok=True)
plt.savefig("data/similarity_heatmap.png")
plt.close()
