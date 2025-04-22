import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

import os
import json
import pubchempy as pcp

CACHE_PATH = "data/smiles_cache.json"

# Load existing cache if it exists
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "r") as f:
        SMILES_CACHE = json.load(f)
else:
    SMILES_CACHE = {}

def save_cache():
    with open(CACHE_PATH, "w") as f:
        json.dump(SMILES_CACHE, f, indent=2)

def get_smiles_from_name(ingredient):
    if ingredient in SMILES_CACHE:
        return SMILES_CACHE[ingredient]  # Could be None if not found

    try:
        compound = pcp.get_compounds(ingredient, 'name')
        if compound and compound[0].canonical_smiles:
            smiles = compound[0].canonical_smiles
            SMILES_CACHE[ingredient] = smiles
            save_cache()
            return smiles
    except Exception:
        pass

    # Cache even if not found, to avoid refetching
    SMILES_CACHE[ingredient] = None
    save_cache()
    return None


def smiles_to_fingerprint(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    except Exception:
        return None
    return None

def compute_tanimoto(fp1, fp2):
    if fp1 and fp2:
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    return 0.0

def fingerprint_similarity(fps1, fps2):
    if not fps1 or not fps2:
        return 0.0
    sims = [compute_tanimoto(fp1, fp2) for fp1 in fps1 for fp2 in fps2]
    return sum(sims) / len(sims) if sims else 0.0
