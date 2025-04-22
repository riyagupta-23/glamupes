# src/normalize_ingredients.py

import re

# Basic normalization dictionary (expand later)
NORMALIZATION_DICT = {
    "aqua": "water",
    "water (aqua)": "water",
    "glycerol": "glycerin",
    "butyrospermum parkii": "shea butter",
    "tocopherol": "vitamin e",
    "niacin": "niacinamide",
    "panthenol": "provitamin b5",
    "ascorbic acid": "vitamin c",
    "sodium ascorbyl phosphate": "vitamin c",
    "retinol": "vitamin a",
    "retinyl palmitate": "vitamin a",
    "ethylhexyl methoxycinnamate": "octinoxate",
    "butyl methoxydibenzoylmethane": "avobenzone",
    "dimethicone": "silicone",
    "cyclopentasiloxane": "silicone",
    "peg-100 stearate": "emulsifier",
    "glyceryl stearate": "emulsifier",
    "cocos nucifera oil": "coconut oil",
    "coconut oil": "coconut oil",
    "helianthus annuus seed oil": "sunflower oil",
    "sunflower seed oil": "sunflower oil",
    "simmondsia chinensis seed oil": "jojoba oil",
    "jojoba seed oil": "jojoba oil",
    "phenoxyethanol": "preservative",
    "ethylhexylglycerin": "preservative booster",
    "benzyl alcohol": "preservative",
    "sodium hyaluronate": "hyaluronic acid",
    "hyaluronic acid": "hyaluronic acid",
    "caprylic/capric triglyceride": "emollient",
    "isopropyl myristate": "emollient",
    "squalane": "squalane",
    "squalene": "squalene",
    "allantoin": "soothing agent",
    "zinc oxide": "mineral sunscreen",
    "titanium dioxide": "mineral sunscreen",
}

def normalize_ingredient(ingredient):
    # Clean non-alphabetic chars, keep things like "PEG-40"
    cleaned = re.sub(r"[^a-zA-Z0-9\- ]", "", ingredient).strip()
    cleaned = cleaned.lower()
    return NORMALIZATION_DICT.get(cleaned, cleaned)

def normalize_ingredient_list(ingredient_list):
    return [normalize_ingredient(i) for i in ingredient_list]
