from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load model globally so it's only loaded once
model = SentenceTransformer("all-MiniLM-L6-v2")

def compute_embeddings(ingredient_lists):
    texts = [" ".join(ing_list) for ing_list in ingredient_lists]
    return model.encode(texts, show_progress_bar=True)

def find_top_similar(embeddings, index, top_n=5):
    similarities = cosine_similarity([embeddings[index]], embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][1:top_n+1]
    return top_indices, similarities[top_indices]
