
from pathlib import Path

import pandas as pd
from transformers import pipeline

MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
MODEL_REVISION = "8f6f4e3a8f70be4b65d3a4a8762b6d781cda240d"

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data" / "raw" / "reviews.csv"
OUTPUT = ROOT / "data" / "processed" / "reviews_with_sentiment.csv"


def to_band(stars: int) -> str:
    return "Negativo" if stars <= 2 else "Neutral" if stars == 3 else "Positivo"


def load_model():
    return pipeline("sentiment-analysis", model=MODEL_NAME, revision=MODEL_REVISION,
                    truncation=True, device=-1)


def predict(df: pd.DataFrame, clf) -> pd.DataFrame:
    df = df.copy()
    df["review_text"] = df["review_text"].str.strip()
    preds = clf(df["review_text"].tolist(), batch_size=32)
    df["pred_stars"] = [int(p["label"][0]) for p in preds]
    df["pred_score"] = [round(p["score"], 4) for p in preds]
    df["pred_band"] = df["pred_stars"].apply(to_band)
    df["true_band"] = df["rating"].apply(to_band)
    return df


def main():
    df = pd.read_csv(INPUT)
    clf = load_model()  # una sola vez
    out = predict(df, clf)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT, index=False)

    print(f"Guardado {len(out)} reseñas en {OUTPUT}")
    print(out["pred_band"].value_counts(normalize=True).mul(100).round(1))


if __name__ == "__main__":
    main()