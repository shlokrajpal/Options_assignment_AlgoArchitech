import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from POI_mapping import POI
from entry_trigger import get_entry_signal
from options_data_processing import get_chain_snapshot, load_chain
from options_execution import select_strike, StrikeSelection

_SPOT_ENV = {
    "1min": "NIFTY_1min_path",
    "5min": "NIFTY_5min_path",
    "1hr": "NIFTY_1hr_path",
}


DEFAULT_CONFIG = {
    "ltf_timeframe": "5min",
    "htf_timeframe": "1hr",
    "starting_capital": 10_000_000.0,
    "delta_target": 0.6,
    "risk_budget": 50_000.0,
    "lot_size": 50,              
    "min_dte": 0,
    "max_lots": 5,
    "hv_window_days": 20,
    "iv_hv_threshold": 1.2,
    "slippage_pct": 0.025,        
    "fees_per_unit": 0.0,         
    "orrr_threshold": 2.0,
    "square_off_time": "15:20",   
    "processed_chain_dir": os.getenv("PROCESSED_CHAIN_DIR"),
}


@dataclass
class TransactionCosts:

    slippage_pct: float = 0.005                 # vs quoted premium, each fill
    brokerage_per_leg: float = 20.0            # flat, per executed leg (Rs)
    exchange_txn_pct: float = 0.035 / 100    # NSE F&O flat txn charge

    def fill_price(self, quoted: float, side: Literal["buy", "sell"]) -> float:
        adj = quoted * self.slippage_pct
        return quoted + adj if side == "buy" else max(0.0, quoted - adj)

    def fees(self, fill: float, qty: int, side: Literal["buy", "sell"]) -> dict:
        turnover = fill * qty
        exch_txn = turnover * self.exchange_txn_pct
        total = self.brokerage_per_leg + exch_txn
        return {
            "brokerage": self.brokerage_per_leg, 
            "exchange_txn": exch_txn,
            "total": total,
        }


@dataclass
class Leg:
    strike: int
    right: str                    # 'CE' / 'PE'
    side: Literal["buy", "sell"]
    qty: int                      # lots * lot_size
    entry_quote: float            # raw chain close at entry
    entry_fill: float             # slippage-adjusted fill
    entry_fees: float


@dataclass
class OpenPosition:
    entry_time: pd.Timestamp
    entry_spot: float
    direction: str                 # 'uptrend' / 'downtrend'
    strategy: str                  # 'naked_long' / 'vertical_spread'
    expiry: date
    lots: int
    lot_size: int
    legs: list[Leg]
    tp_spot: float
    sl_spot: float
    chain: pd.DataFrame
    selection: StrikeSelection


def load_spot(timeframe: str) -> pd.DataFrame:
    if timeframe not in _SPOT_ENV:
        raise ValueError(f"Unknown timeframe '{timeframe}', expected one of {list(_SPOT_ENV)}")
    path = os.getenv(_SPOT_ENV[timeframe])
    if not path:
        raise EnvironmentError(
            f"Env var {_SPOT_ENV[timeframe]} not set (needed for '{timeframe}' spot data)"
        )
    df = pd.read_csv(path, parse_dates=["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)

def load_catalog(processed_chain_dir: str | Path) -> pd.DataFrame:
    path = Path(processed_chain_dir) / "index.parquet"
    cat = pd.read_parquet(path)
    cat["trade_date"] = pd.to_datetime(cat["trade_date"]).dt.date
    cat["expiry"] = pd.to_datetime(cat["expiry"]).dt.date
    return cat

def select_expiry(catalog: pd.DataFrame, trade_date_: date, min_dte: int = 0) -> Optional[date]:
    day = catalog[(catalog["trade_date"] == trade_date_) & (catalog["dte"] >= min_dte)]
    if day.empty:
        return None
    return day.sort_values("dte").iloc[0]["expiry"]

def open_position(signal: dict, catalog: pd.DataFrame, config: dict,
                   costs: TransactionCosts) -> Optional[OpenPosition]:
    trade_date_ = pd.Timestamp(signal["timestamp"]).date()
    expiry = select_expiry(catalog, trade_date_, min_dte=config.get("min_dte", 0))
    if expiry is None:
        return None  

    try:
        chain = load_chain(trade_date_, expiry, config["processed_chain_dir"])
    except FileNotFoundError:
        return None

    strike_cfg = {
        "delta_target": config["delta_target"],
        "risk_budget": config["risk_budget"],
        "expiry_datetime": pd.Timestamp(expiry) + pd.Timedelta(hours=15, minutes=30),
        "lot_size": config.get("lot_size", 1),
        "max_lots": config.get("max_lots", 5),
        **{k: v for k, v in config.items() if k in (
            "hv_window_days", "iv_hv_threshold", "orrr_threshold",
            "slippage_pct", "fees_per_unit",
        )},
    }

    try:
        selection = select_strike(signal, chain, strike_cfg)
    except (ValueError, KeyError):
        return None 

    if selection.lots <= 0:
        return None  

    lot_size = strike_cfg["lot_size"]
    qty = selection.lots * lot_size

    try:
        snapshot = get_chain_snapshot(chain, signal["timestamp"], signal["entry_price"], prune=10)
        long_quote = float(snapshot.loc[(selection.K_star, selection.right), "close"])
    except KeyError:
        return None

    fill = costs.fill_price(long_quote, "buy")
    fees = costs.fees(fill, qty, "buy")["total"]
    legs = [Leg(selection.K_star, selection.right, "buy", qty, long_quote, fill, fees)]

    strategy = "naked_long"
    if selection.route == "spread":
        try:
            short_quote = float(snapshot.loc[(selection.K_short, selection.right), "close"])
        except KeyError:
            return None 
        strategy = "vertical_spread"
        fill_s = costs.fill_price(short_quote, "sell")
        fees_s = costs.fees(fill_s, qty, "sell")["total"]
        legs.append(Leg(selection.K_short, selection.right, "sell", qty, short_quote, fill_s, fees_s))

    round_trip_cost = 0.0
    for leg in legs:
        exit_side = "sell" if leg.side == "buy" else "buy"
        est_exit_fill = costs.fill_price(leg.entry_quote, exit_side)
        est_exit_fees = costs.fees(est_exit_fill, leg.qty, exit_side)["total"]
        slippage = abs(leg.entry_fill - leg.entry_quote) * leg.qty + abs(leg.entry_quote - est_exit_fill) * leg.qty
        round_trip_cost += leg.entry_fees + est_exit_fees + slippage

    cost_per_unit = round_trip_cost / qty
    friction_hurdle_points = cost_per_unit / max(abs(selection.delta_k_star), 1e-6)
    distance_to_tp = abs(signal["take_profit"] - signal["entry_price"])
    if distance_to_tp < 1.01 * friction_hurdle_points:
        return None

    return OpenPosition(
        entry_time=signal["timestamp"], entry_spot=signal["entry_price"],
        direction=signal["direction"], strategy=strategy, expiry=expiry,
        lots=selection.lots, lot_size=lot_size, legs=legs,
        tp_spot=signal["take_profit"], sl_spot=signal["stop_loss"],
        chain=chain, selection=selection,
    )


def check_exit(position: OpenPosition, ltf_df: pd.DataFrame, curr_idx: int,
               entry_idx: int, curr_time: pd.Timestamp, config: dict) -> Optional[dict]:
    bar = ltf_df.iloc[curr_idx]

    if position.direction == "uptrend":
        hit_tp = bar["high"] >= position.tp_spot
        hit_sl = bar["low"] <= position.sl_spot
    else:
        hit_tp = bar["low"] <= position.tp_spot
        hit_sl = bar["high"] >= position.sl_spot

    open_h, open_m = (int(x) for x in config.get("square_off_time", "15:20").split(":"))
    square_off_cutoff = pd.Timestamp(position.expiry) + pd.Timedelta(hours=open_h, minutes=open_m)
    forced_expiry = curr_time >= square_off_cutoff

    if hit_sl:
        return {"time": curr_time, "reason": "stop_loss", "spot_price": position.sl_spot}
    if hit_tp:
        return {"time": curr_time, "reason": "take_profit", "spot_price": position.tp_spot}
    if forced_expiry:
        return {"time": curr_time, "reason": "expiry_square_off", "spot_price": float(bar["close"])}
    return None


def close_position(position: OpenPosition, exit_info: dict, costs: TransactionCosts) -> dict:
    try:
        snapshot = get_chain_snapshot(position.chain, exit_info["time"], exit_info["spot_price"], prune=10)
    except IndexError:
        snapshot = None

    gross_pnl = 0.0
    net_pnl = 0.0
    total_fees = 0.0
    slippage_cost = 0.0
    exit_by_side = {}

    for leg in position.legs:
        try:
            exit_quote = float(snapshot.loc[(leg.strike, leg.right), "close"]) if snapshot is not None else 0.0
        except KeyError:
            exit_quote = 0.0 

        exit_side: Literal["buy", "sell"] = "sell" if leg.side == "buy" else "buy"
        exit_fill = costs.fill_price(exit_quote, exit_side)
        exit_fees = costs.fees(exit_fill, leg.qty, exit_side)["total"]

        if leg.side == "buy":
            quote_pnl = (exit_quote - leg.entry_quote) * leg.qty
            fill_pnl = (exit_fill - leg.entry_fill) * leg.qty
        else:
            quote_pnl = (leg.entry_quote - exit_quote) * leg.qty
            fill_pnl = (leg.entry_fill - exit_fill) * leg.qty

        leg_fees = leg.entry_fees + exit_fees
        gross_pnl += quote_pnl
        net_pnl += fill_pnl - leg_fees
        total_fees += leg_fees
        slippage_cost += quote_pnl - fill_pnl
        exit_by_side[leg.side] = exit_fill

    long_leg = next(l for l in position.legs if l.side == "buy")
    short_leg = next((l for l in position.legs if l.side == "sell"), None)

    return {
        "entry_time": position.entry_time,
        "exit_time": exit_info["time"],
        "entry_price": position.entry_spot,
        "exit_price": exit_info["spot_price"],
        "strategy": position.strategy,
        "direction": position.direction,
        "right": long_leg.right,
        "long_strike": long_leg.strike,
        "short_strike": short_leg.strike if short_leg else None,
        "lots": position.lots,
        "lot_size": position.lot_size,
        "long_entry_premium": long_leg.entry_quote,
        "long_exit_premium": exit_by_side.get("buy"),
        "short_entry_premium": short_leg.entry_quote if short_leg else None,
        "short_exit_premium": exit_by_side.get("sell"),
        "gross_pnl": round(gross_pnl, 2),
        "slippage_cost": round(slippage_cost, 2),
        "fees": round(total_fees, 2),
        "net_pnl": round(net_pnl, 2),
        "exit_reason": exit_info["reason"],
        "expiry": position.expiry,
        "orrr_at_entry": position.selection.orrr_final,
        "route_reason": position.selection.route,
    }


TRADESHEET_COLUMNS = [
    "entry_time", "exit_time", "entry_price", "exit_price", "strategy", "direction",
    "right", "long_strike", "short_strike", "lots", "lot_size",
    "long_entry_premium", "long_exit_premium", "short_entry_premium", "short_exit_premium",
    "gross_pnl", "slippage_cost", "fees", "net_pnl", "total_capital", "exit_reason", "expiry",
    "orrr_at_entry", "route_reason",
]


def run_backtest(config: Optional[dict] = None) -> pd.DataFrame:
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    if not cfg["processed_chain_dir"]:
        raise EnvironmentError(
            "Set PROCESSED_CHAIN_DIR env var or pass config['processed_chain_dir'] "
            "(output directory of options_data_processing.build_dataset)"
        )

    ltf_df = load_spot(cfg["ltf_timeframe"])
    htf_df = load_spot(cfg["htf_timeframe"])
    catalog = load_catalog(cfg["processed_chain_dir"])
    costs = cfg.get("costs") or TransactionCosts(**cfg.get("cost_overrides", {}))

    ltf_indexed = ltf_df.set_index("datetime", drop=False)
    htf_indexed = htf_df.set_index("datetime", drop=False)

    cache: dict = {}
    in_position = False
    position: Optional[OpenPosition] = None
    entry_idx: Optional[int] = None
    trades: list[dict] = []
    total_capital = float(cfg.get("starting_capital", 10_000_000.0))

    for i, curr_time in enumerate(ltf_df["datetime"]):
        if in_position:
            exit_info = check_exit(position, ltf_df, i, entry_idx, curr_time, cfg)
            if exit_info is not None:
                closed_trade = close_position(position, exit_info, costs)
                total_capital += float(closed_trade["net_pnl"])
                closed_trade["total_capital"] = round(total_capital, 2)
                trades.append(closed_trade)
                in_position = False
                position = None
                entry_idx = None
                cache["poi"] = []  # force a fresh POI scan once flat again
            continue

        signal = get_entry_signal(curr_time, i, ltf_indexed, htf_indexed, cache, in_position)
        if signal is None:
            continue

        new_position = open_position(signal, catalog, cfg, costs)
        if new_position is not None:
            position = new_position
            in_position = True
            entry_idx = i
        else:
            cache["poi"] = []

    df = pd.DataFrame(trades)
    if not df.empty:
        df = df[[c for c in TRADESHEET_COLUMNS if c in df.columns]]
    return df


def summarize_tradesheet(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"trades": 0}
    wins = df["net_pnl"] > 0
    equity = df["total_capital"] if "total_capital" in df.columns else df["net_pnl"].cumsum()
    drawdown = equity - equity.cummax()
    gross_profit = df.loc[wins, "net_pnl"].sum()
    gross_loss = -df.loc[~wins, "net_pnl"].sum()
    return {
        "trades": int(len(df)),
        "win_rate": round(float(wins.mean()), 4),
        "total_gross_pnl": round(float(df["gross_pnl"].sum()), 2),
        "total_fees": round(float(df["fees"].sum()), 2),
        "total_slippage_cost": round(float(df["slippage_cost"].sum()), 2),
        "total_net_pnl": round(float(df["net_pnl"].sum()), 2),
        "ending_capital": round(float(equity.iloc[-1]), 2),
        "avg_net_pnl": round(float(df["net_pnl"].mean()), 2),
        "profit_factor": round(float(gross_profit / gross_loss), 2) if gross_loss > 0 else float("inf"),
        "max_drawdown": round(float(drawdown.min()), 2),
        "naked_long_trades": int((df["strategy"] == "naked_long").sum()),
        "vertical_spread_trades": int((df["strategy"] == "vertical_spread").sum()),
    }


if __name__ == "__main__":
    tradesheet = run_backtest()
    out_path = os.getenv("TRADESHEET_OUTPUT_PATH")
    tradesheet.to_csv(out_path, index=False)
    print(f"Wrote {len(tradesheet)} trades -> {out_path}")
    print(summarize_tradesheet(tradesheet))
