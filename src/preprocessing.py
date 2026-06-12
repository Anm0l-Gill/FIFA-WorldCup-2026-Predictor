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

def add_rolling_stats(df):
    df = df.sort_values("date").reset_index(drop=True)

    def rolling(df, team_col, score_col, concede_col, suffix):
        stats = []
        for team in df[team_col].unique():
            mask    = df[team_col] == team
            team_df = df[mask].copy()
            team_df[f"goals_scored_{suffix}"]  = team_df[score_col].shift(1).rolling(5, min_periods=1).mean()
            team_df[f"goals_conceded_{suffix}"] = team_df[concede_col].shift(1).rolling(5, min_periods=1).mean()
            stats.append(team_df[["date", team_col,
                                   f"goals_scored_{suffix}",
                                   f"goals_conceded_{suffix}"]])
        return pd.concat(stats).sort_values("date")

    home_stats = rolling(df, "home_team", "home_score", "away_score", "home")
    away_stats = rolling(df, "away_team", "away_score", "home_score", "away")

    df = df.merge(home_stats, on=["date", "home_team"], how="left")
    df = df.merge(away_stats, on=["date", "away_team"], how="left")
    return df

if __name__ == "__main__":
    results, rankings, elo = load_data()

    results  = fix_names(results,  ["home_team", "away_team"])
    rankings = fix_names(rankings, ["country_full"])
    elo      = fix_names(elo,      ["team"])

    results = filter_cycle(results)
    results = add_result(results)

    print("merging rankings...")
    results = merge_rankings(results, rankings)

    print("merging elo...")
    results = merge_elo(results, elo)

    results = add_features(results)
    results = add_rolling_stats(results)

    # full dataset — keep all matches that have rankings (used by Poisson + ML with rank features)
    full = results.dropna(subset=["home_rank", "away_rank"])
    print(f"full dataset: {len(full)} matches")
    full.to_csv(PROCESSED / "match_features.csv", index=False)

    # model-ready with all features including elo (smaller, used as secondary check)
    model_cols = ["rank_diff", "point_diff", "elo_diff", "is_friendly",
                  "goals_scored_home", "goals_conceded_home",
                  "goals_scored_away", "goals_conceded_away", "result"]

    model_full = full.dropna(subset=["rank_diff", "point_diff",
                                      "goals_scored_home", "goals_scored_away"])[model_cols]
    print(f"model dataset (no elo required): {len(model_full)} matches")
    model_full.to_csv(PROCESSED / "model_features.csv", index=False)

    # elo subset — only WC teams, used to validate elo_diff matters
    model_elo = full.dropna(subset=["home_elo", "away_elo"])[model_cols]
    print(f"elo subset: {len(model_elo)} matches")
    model_elo.to_csv(PROCESSED / "model_features_elo.csv", index=False)

    print("done")