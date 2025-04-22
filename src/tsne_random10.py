import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import random

from normalize_ingredients import normalize_ingredient_list
from load_data import load_product_data
from nlp_similarity import compute_embeddings, find_top_similar

# Load and preprocess
df = load_product_data("data/makeup-products-list.csv")
df["normalized_ingredients"] = df["ingredients"].apply(normalize_ingredient_list)
df["normalized_ingredients"] = df["normalized_ingredients"].apply(lambda x: list(dict.fromkeys(x)))

# Compute NLP embeddings
embeddings = compute_embeddings(df["normalized_ingredients"].tolist())

# Run t-SNE on all embeddings once
reduced = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(np.array(embeddings))

# Make sure output directory exists
output_dir = "tsne_graphs"
os.makedirs(output_dir, exist_ok=True)

# Select 10 random unique indices
random_indices = random.sample(range(len(df)), 10)

# Generate and save graphs
for index in random_indices:
    top_indices, _ = find_top_similar(embeddings, index, top_n=5)
    highlight_indices = [index] + list(top_indices)

    plt.figure(figsize=(12, 8))
    for i, (x, y) in enumerate(reduced):
        if i in highlight_indices:
            plt.scatter(x, y, c='red', s=100, edgecolors='k')
            plt.annotate(df.iloc[i]["glamupe_name"][:25], (x, y), fontsize=9)
        else:
            plt.scatter(x, y, c='gray', s=40, alpha=0.5)

    title = df.iloc[index]['glamupe_name'][:40].replace("/", "_")
    plt.title(f"t-SNE: {title}")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"tsne_product_{index}.png"))
    plt.close()
