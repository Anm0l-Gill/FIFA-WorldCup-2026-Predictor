import pandas as pd
import numpy as np
from pathlib import Path

ROOT      = Path(__file__).resolve().parent.parent
RAW       = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

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
    elo      = pd.read_csv(RAW / "teams_elo.csv")
    return results, rankings, elo

def fix_names(df, cols):
    for col in cols:
        df[col] = df[col].replace(NAME_FIXES)
    return df

def filter_cycle(df, start="2022-12-19"):
    return df[df["date"] >= start].reset_index(drop=True)

def add_result(df):
    df["result"] = np.where(
        df["home_score"] > df["away_score"], 1,
        np.where(df["home_score"] == df["away_score"], 0, -1)
    )
    return df

def merge_rankings(results, rankings):
    rankings = rankings[["rank_date", "country_full", "rank", "total_points"]].copy()

    # rankings are published monthly so we forward-fill to daily
    teams      = rankings["country_full"].unique()
    date_range = pd.date_range(rankings["rank_date"].min(), results["date"].max(), freq="D")

    filled = []
    for team in teams:
        team_df = (rankings[rankings["country_full"] == team]
                   .set_index("rank_date")
                   .reindex(date_range)
                   .ffill()
                   .reset_index()
                   .rename(columns={"index": "date"}))
        team_df["country_full"] = team
        filled.append(team_df)

    daily = pd.concat(filled, ignore_index=True)

    results = results.merge(
        daily.rename(columns={"country_full": "home_team",
                               "rank":         "home_rank",
                               "total_points": "home_points"}),
        on=["date", "home_team"], how="left"
    )
    results = results.merge(
        daily.rename(columns={"country_full": "away_team",
                               "rank":         "away_rank",
                               "total_points": "away_points"}),
        on=["date", "away_team"], how="left"
    )
    return results

def merge_elo(results, elo):
    elo = elo[["team", "elo"]].copy()
    results = results.merge(
        elo.rename(columns={"team": "home_team", "elo": "home_elo"}),
        on="home_team", how="left"
    )
    results = results.merge(
        elo.rename(columns={"team": "away_team", "elo": "away_elo"}),
        on="away_team", how="left"
    )
    return results

def add_features(df):
    df["rank_diff"]   = df["home_rank"]   - df["away_rank"]
    df["point_diff"]  = df["home_points"] - df["away_points"]
    df["elo_diff"]    = df["home_elo"]    - df["away_elo"]
    df["is_friendly"] = (df["tournament"] == "Friendly").astype(int)
    return df

if __name__ == "__main__":
    results, rankings, elo = load_data()

    results  = fix_names(results,  ["home_team", "away_team"])
    rankings = fix_names(rankings, ["country_full"])
    elo      = fix_names(elo,      ["team"])

    results = filter_cycle(results)
    results = add_result(results)

    print("merging rankings (this takes ~20 seconds)...")
    results = merge_rankings(results, rankings)

    print("merging elo...")
    results = merge_elo(results, elo)

    results = add_features(results)

    before  = len(results)
    results = results.dropna(subset=["home_rank", "away_rank"])
    print(f"dropped {before - len(results)} rows with missing rankings")
    print(f"final dataset: {len(results)} matches")
    print(results[["date", "home_team", "away_team", "home_rank", "away_rank",
                   "home_elo", "away_elo", "rank_diff", "elo_diff", "result"]].head())

    results.to_csv(PROCESSED / "match_features.csv", index=False)
    print("saved to data/processed/match_features.csv")