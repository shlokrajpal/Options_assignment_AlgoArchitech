import os
import re
from pathlib import Path
from datetime import date
from collections import defaultdict

import pandas as pd
from dateutil import parser as dtparser
from dotenv import load_dotenv

load_dotenv()

NIFTY_1MIN_PATH = os.getenv("NIFTY_1min_path")
FILENAME_RE = re.compile(r"NIFTY-(.+)-(.+)\.csv$")

def parse_filename(filename: str) -> tuple[date, date]:
    m = FILENAME_RE.match(filename)
    if not m:
        raise ValueError(f"Filename doesn't match NIFTY-<expiry>-<date>.csv: {filename}")
    expiry_str, date_str = m.groups()
    expiry_date = dtparser.parse(expiry_str, dayfirst=True).date()
    trade_date = dtparser.parse(date_str, dayfirst=True).date()
    return expiry_date, trade_date

def load_nifty_spot(csv_path: str) -> pd.Series:
    """Loads Nifty 1-min data and returns a series of close prices indexed by datetime."""
    df = pd.read_csv(csv_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    return df["close"]

def pivot_chain(csv_path: Path, trade_date: date) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        usecols=["datetime", "strike_price", "right", "open", "high", "low", "close", "open_interest"],
    )
    df["datetime"] = pd.to_datetime(trade_date.isoformat() + " " + df["datetime"])

    wide = df.pivot_table(
        index="datetime",
        columns=["strike_price", "right"],
        values=["open", "high", "low", "close", "open_interest"],
        aggfunc="last",
    )
    wide.columns = [f"{int(strike)}_{right}_{field}" for field, strike, right in wide.columns]
    wide = wide.sort_index(axis=0).sort_index(axis=1)
    return wide

def build_dataset(raw_dir: Path, processed_dir: Path) -> pd.DataFrame:
    raw_dir, processed_dir = Path(raw_dir), Path(processed_dir)
    chain_dir = processed_dir / "chain"
    chain_dir.mkdir(parents=True, exist_ok=True)

    catalog_rows = []
    for csv_path in sorted(raw_dir.glob("*/NIFTY-*.csv")):
        try:
            expiry_date, trade_date = parse_filename(csv_path.name)
        except ValueError as e:
            print(f"SKIP {csv_path}: {e}")
            continue

        wide = pivot_chain(csv_path, trade_date)

        out_dir = chain_dir / trade_date.isoformat()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"NIFTY-{expiry_date.isoformat()}.parquet"
        wide.to_parquet(out_path)

        strikes = sorted({int(c.split("_")[0]) for c in wide.columns})
        catalog_rows.append(
            {
                "trade_date": trade_date,
                "expiry": expiry_date,
                "dte": (expiry_date - trade_date).days,
                "n_strikes": len(strikes),
                "strike_min": strikes[0],
                "strike_max": strikes[-1],
                "path": str(out_path),
            }
        )

    catalog = pd.DataFrame(catalog_rows).sort_values(["trade_date", "expiry"]).reset_index(drop=True)
    catalog.to_parquet(processed_dir / "index.parquet")
    catalog.to_csv(processed_dir / "index.csv", index=False)
    return catalog

def load_chain(trade_date: date, expiry_date: date, processed_dir: Path) -> pd.DataFrame:
    path = Path(processed_dir) / "chain" / trade_date.isoformat() / f"NIFTY-{expiry_date.isoformat()}.parquet"
    return pd.read_parquet(path)


def get_chain_snapshot(chain: pd.DataFrame, at_time, nifty_spot: float, prune: int = None) -> pd.DataFrame:
    row = chain.loc[chain.index <= pd.Timestamp(at_time)].iloc[-1]
    records: dict = {}
    for col, val in row.items():
        strike, right, field = col.split("_")
        records.setdefault((int(strike), right), {})[field] = val
    
    snap = pd.DataFrame.from_dict(records, orient="index")
    snap.index = pd.MultiIndex.from_tuples(snap.index, names=["strike", "right"])
    snap = snap.sort_index()

    if prune is not None:
        atm_strike = round(nifty_spot / 50.0) * 50
        unique_strikes = sorted(list(set(snap.index.get_level_values("strike"))))
        if atm_strike in unique_strikes:
            atm_idx = unique_strikes.index(atm_strike)
        else:
            atm_idx = min(range(len(unique_strikes)), key=lambda i: abs(unique_strikes[i] - atm_strike))
        start_idx = max(0, atm_idx - prune)
        end_idx = min(len(unique_strikes), atm_idx + prune + 1)
        kept_strikes = unique_strikes[start_idx:end_idx]
        snap = snap.loc[snap.index.get_level_values("strike").isin(kept_strikes)]
        
    return snap

def get_strike_series(chain: pd.DataFrame, strike: int, right: str) -> pd.DataFrame:
    cols = [f"{strike}_{right}_{f}" for f in ["open", "high", "low", "close", "open_interest"]]
    out = chain[cols].copy()
    out.columns = ["open", "high", "low", "close", "open_interest"]
    return out


def analyze_coverage_stats(raw_dir: Path, nifty_spot_series: pd.Series):
    print("Analyzing coverage statistics. This may take a few moments...")
    oi_sums = defaultdict(float)
    oi_counts = defaultdict(int)

    for csv_path in raw_dir.glob("*/NIFTY-*.csv"):
        try:
            _, trade_date = parse_filename(csv_path.name)
        except ValueError:
            continue
        
        df = pd.read_csv(csv_path, usecols=["datetime", "strike_price", "right", "open_interest"])
        df["datetime"] = pd.to_datetime(trade_date.isoformat() + " " + df["datetime"])
        df = df.merge(nifty_spot_series.rename("spot"), left_on="datetime", right_index=True, how="inner")
        df["atm"] = (df["spot"] / 50.0).round() * 50
        df["strike_dist"] = ((df["strike_price"] - df["atm"]) / 50.0).astype(int)

        # Aggregate OI by distance and right (CE/PE)
        agg = df.groupby(["strike_dist", "right"])["open_interest"].agg(["sum", "count"])
        
        for (dist, right), row in agg.iterrows():
            oi_sums[(dist, right)] += row["sum"]
            oi_counts[(dist, right)] += row["count"]

    print("\n--- Average Open Interest by Distance from ATM ---")
    print(f"{'Distance (Strikes)':<20} | {'Call (CE) Avg OI':<20} | {'Put (PE) Avg OI':<20}")
    print("-" * 65)
    
    all_distances = sorted(list(set(k[0] for k in oi_sums.keys())))
    for dist in all_distances:
        if abs(dist) > 20: 
            continue 
            
        ce_avg = oi_sums.get((dist, "CE"), 0) / oi_counts.get((dist, "CE"), 1)
        pe_avg = oi_sums.get((dist, "PE"), 0) / oi_counts.get((dist, "PE"), 1)
        
        print(f"{dist:<20} | {ce_avg:<20.2f} | {pe_avg:<20.2f}")


if __name__ == "__main__":
    raw = Path(r"D:\Options_assignment_AlgoArchitech\data") 
    out = Path(r"D:\Options_assignment_AlgoArchitech\options_data")
    
    print(f"Loading Nifty 1-min spot data from {NIFTY_1MIN_PATH}...")
    nifty_spot = load_nifty_spot(NIFTY_1MIN_PATH)
    
    analyze_coverage_stats(raw, nifty_spot)
    
    cat = build_dataset(raw, out)
    print(f"Built {len(cat)} (date, expiry) chain files -> {out}")