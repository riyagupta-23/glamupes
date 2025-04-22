import pandas as pd

def load_product_data(filepath):
    df = pd.read_csv(
        filepath,
        header=0,
        names=["glamupe_id", "glamupe_name", "glamupe_ingredients", "glamupe_photo_url", "label"],
        quotechar='"',
        encoding="utf-8",
        on_bad_lines='skip'
    )

    # Only keep name + ingredients
    df = df[["glamupe_id", "glamupe_name", "glamupe_ingredients"]]
    df.dropna(subset=["glamupe_name", "glamupe_ingredients"], inplace=True)

    # Parse ingredients into list
    df["ingredients"] = df["glamupe_ingredients"].apply(
        lambda x: [i.strip().lower() for i in x.split(",")]
    )

    return df


