import pandas as pd

ML_PATH = "final_dataset_ml_ready_numeric_plus_extended.csv"
CLEAN_PATH = "final_dataset_all_cleaned.csv"
OUT_PATH = "final_dataset_ml_ready_numeric_plus_extended_with_title.csv"


def main():
    ml = pd.read_csv(ML_PATH)
    base = pd.read_csv(CLEAN_PATH, usecols=["title"])

    if len(ml) != len(base):
        raise ValueError("Row count mismatch between ML and cleaned datasets.")

    ml.insert(0, "title", base["title"])
    ml.to_csv(OUT_PATH, index=False)
    print(f"Saved merged dataset with title to {OUT_PATH}")


if __name__ == "__main__":
    main() 