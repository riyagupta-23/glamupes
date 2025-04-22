from sklearn.metrics.pairwise import cosine_similarity
from src.load_data import load_product_data
from src.normalize_ingredients import normalize_ingredient_list
from src.nlp_similarity import compute_embeddings, find_top_similar
from src.chem_utils import (
    get_smiles_from_name,
    smiles_to_fingerprint,
    fingerprint_similarity
)
import numpy as np

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os


def plot_tsne_highlighted(embeddings, labels, highlight_indices, title="t-SNE of All Products"):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    reduced = tsne.fit_transform(np.array(embeddings))

    plt.figure(figsize=(12, 8))
    for i, (x, y) in enumerate(reduced):
        if i in highlight_indices:
            plt.scatter(x, y, c='red', s=100, label='Target/Dupes' if i == highlight_indices[0] else "", edgecolors='k')
            plt.annotate(labels[i][:25], (x, y), fontsize=9)
        else:
            plt.scatter(x, y, c='gray', s=40, alpha=0.5)

    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(True)
    plt.tight_layout()
    handles, labels_ = plt.gca().get_legend_handles_labels()
    if 'Target/Dupes' in labels_:
        plt.legend()
    plt.show()




if __name__ == "__main__":


    df = load_product_data("data/makeup-products-list.csv")

    df["ingredients"] = df["glamupe_ingredients"].apply(lambda x: [i.strip().lower() for i in x.split(",")])
    df["normalized_ingredients"] = df["ingredients"].apply(normalize_ingredient_list)
    df["normalized_ingredients"] = df["normalized_ingredients"].apply(lambda x: list(dict.fromkeys(x)))

    # NLP embedding
    embeddings = compute_embeddings(df["normalized_ingredients"])

    # Example: Get top 5 NLP-based dupes
    index = 0
    top_indices, scores = find_top_similar(embeddings, index, top_n=5)


    print(f"\nTop NLP-based dupes for: {df.iloc[index]['glamupe_name']}")
    for i, idx in enumerate(top_indices):
        print(f"{i+1}. {df.iloc[idx]['glamupe_name']} — score: {scores[i]:.4f}")

# Plot all embeddings and highlight the target product and its dupes
    highlight_indices = [index] + list(top_indices)
    plot_tsne_highlighted(embeddings, df["glamupe_name"].tolist(), highlight_indices)


    # ---------- Move everything below into the main block too ----------
    def convert_ingredients_to_fingerprints(ingredient_list):
        fps = []
        for ing in ingredient_list:
            smiles = get_smiles_from_name(ing)
            if smiles:
                fp = smiles_to_fingerprint(smiles)
                if fp:
                    fps.append(fp)
        return fps

    # Apply to all products
    df["fingerprints"] = df["normalized_ingredients"].apply(convert_ingredients_to_fingerprints)

    # Define it
    def get_top_chemical_dupes(df, target_index=70, top_n=5):
        # --- Combined Similarity (Chemical + NLP) ---
        target_fps = df.iloc[target_index]["fingerprints"]
        chem_sims = df["fingerprints"].apply(lambda fps: fingerprint_similarity(target_fps, fps)).values
        nlp_sims = cosine_similarity([embeddings[target_index]], embeddings)[0]

        alpha = 0.5
        combined_scores = alpha * nlp_sims + (1 - alpha) * chem_sims
        combined_scores[target_index] = -1

        top_combined_indices = np.argsort(combined_scores)[::-1][:top_n]

        print(f"\n🔀 Top HYBRID dupes for: {df.iloc[target_index]['glamupe_name']}")
        for i, idx in enumerate(top_combined_indices, start=1):
            print(f"{i}. {df.iloc[idx]['glamupe_name']}")
            print(f"   🔬 Chemical Score: {chem_sims[idx]:.4f}")
            print(f"   🧠 NLP Score:      {nlp_sims[idx]:.4f}")
            print(f"   ⚖️  Combined:       {combined_scores[idx]:.4f}\n")

    # ✅ Call it AFTER the function definition
    get_top_chemical_dupes(df, target_index=0)
