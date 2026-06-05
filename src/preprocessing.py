import pandas as pd
import numpy as np
from pathlib import Path

ROOT      = Path(__file__).resolve().parent.parent
RAW       = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

# team name differences between the two datasets
NAME_FIXES = {
    "IR Iran":        "Iran",
    "Korea Republic": "South Korea",
    "USA":            "United States",
    "Türkiye":        "Turkey",
    "China PR":       "China",
}

def load_data():
    results  = pd.read_csv(RAW / "results.csv",      parse_dates=["date"])
    rankings = pd.read_csv(RAW / "fifa_ranking.csv", parse_dates=["rank_date"])
    return results, rankings

def fix_names(df, cols):
    for col in cols:
        df[col] = df[col].replace(NAME_FIXES)
    return df

def filter_cycle(df, start="2022-12-19"):
    # only keep matches from after the 2022 WC final
    return df[df["date"] >= start].reset_index(drop=True)

def add_result(df):
    df["result"] = np.where(
        df["home_score"] > df["away_score"], 1,
        np.where(df["home_score"] == df["away_score"], 0, -1)
    )
    return df

if __name__ == "__main__":
    results, rankings = load_data()

    results  = fix_names(results,  ["home_team", "away_team"])
    rankings = fix_names(rankings, ["country_full"])
    results  = filter_cycle(results)
    results  = add_result(results)

    print(f"matches loaded: {len(results)}")
    print(results[["date", "home_team", "away_team", "home_score", "away_score", "result"]].head())

    results.to_csv(PROCESSED / "match_features.csv", index=False)
