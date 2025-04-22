import os
import pandas as pd
import numpy as np
import random

from load_data import load_product_data
from normalize_ingredients import normalize_ingredient_list
from nlp_similarity import compute_embeddings
from chem_utils import get_smiles_from_name, smiles_to_fingerprint, fingerprint_similarity
from sklearn.metrics.pairwise import cosine_similarity

# Load and preprocess
df = load_product_data("data/makeup-products-list.csv")
df["normalized_ingredients"] = df["ingredients"].apply(normalize_ingredient_list)
df["normalized_ingredients"] = df["normalized_ingredients"].apply(lambda x: list(dict.fromkeys(x)))

# Compute NLP embeddings
embeddings = compute_embeddings(df["normalized_ingredients"].tolist())

# Generate fingerprints
def convert_ingredients_to_fingerprints(ingredient_list):
    fps = []
    for ing in ingredient_list:
        smiles = get_smiles_from_name(ing)
        if smiles:
            fp = smiles_to_fingerprint(smiles)
            if fp:
                fps.append(fp)
    return fps

df["fingerprints"] = df["normalized_ingredients"].apply(convert_ingredients_to_fingerprints)

# Sample 30 random products
random.seed(42)
random_indices = random.sample(range(len(df)), 30)

# Create match results table
rows = []
alpha = 0.5

for idx in random_indices:
    target_name = df.iloc[idx]["glamupe_name"]
    target_fps = df.iloc[idx]["fingerprints"]
    target_embed = embeddings[idx]

    chem_sims = df["fingerprints"].apply(lambda fps: fingerprint_similarity(target_fps, fps)).values
    nlp_sims = cosine_similarity([target_embed], embeddings)[0]
    combined = alpha * nlp_sims + (1 - alpha) * chem_sims
    combined[idx] = -1

    best_idx = np.argmax(combined)
    rows.append({
        "Target Product": target_name,
        "Top Match": df.iloc[best_idx]["glamupe_name"],
        "NLP Score": round(nlp_sims[best_idx], 4),
        "Chem Score": round(chem_sims[best_idx], 4),
        "Combined Score": round(combined[best_idx], 4)
    })

# Save table
os.makedirs("data", exist_ok=True)
results_df = pd.DataFrame(rows)
results_df.to_csv("data/table_top_matches.csv", index=False)
