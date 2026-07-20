from dataclasses import dataclass
from math import log, sqrt, floor
from typing import Optional

from dotenv import load_dotenv
import os

load_dotenv()

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq

from options_data_processing import get_chain_snapshot

def get_spotNifty_1hr() -> pd.DataFrame:
    df = pd.read_csv(os.getenv("NIFTY_1hr_path"), parse_dates=["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)
     

@dataclass
class StrikeSelection:
    entry_time: pd.Timestamp
    direction: str                 # 'uptrend'/'downtrend'
    right: str                     # 'CE'/'PE'
    entry_price: float             # S
    take_profit: float
    stop_loss: float
    F: float
    K_star: int                    # naked leg (K*)
    sigma_k_star: float
    delta_k_star: float
    gamma_k_star: float
    theta_k_star: float           
    delta_F_tp: float
    delta_F_sl: float
    expected_reward_naked: float
    expected_risk_naked: float
    orrr_naked: float
    iv_hv_ratio: float
    hv_20: float
    filter1_triggered: bool
    filter3_triggered: bool
    route: str                     # 'naked_long' | 'spread'
    # spread leg
    K_short: Optional[int]
    sigma_k_short: Optional[float]
    delta_k_short: Optional[float]
    gamma_k_short: Optional[float]
    theta_k_short: Optional[float]
    expected_reward_spread: Optional[float]
    expected_risk_spread: Optional[float]
    orrr_spread: Optional[float]
    theta_net: Optional[float]
    orrr_final: float
    lots: int


def compute_T(entry_time: pd.Timestamp, expiry_datetime: pd.Timestamp,
              session_open: str = "09:15", session_close: str = "15:30",
              trading_days_per_year: int = 252) -> float:

    entry_time = pd.Timestamp(entry_time)
    expiry_datetime = pd.Timestamp(expiry_datetime)
    if expiry_datetime <= entry_time:
        return 0.0

    open_h, open_m = (int(x) for x in session_open.split(":"))
    close_h, close_m = (int(x) for x in session_close.split(":"))
    minutes_per_day = (close_h * 60 + close_m) - (open_h * 60 + open_m)

    if entry_time.normalize() == expiry_datetime.normalize():
        minutes_to_expiry = (expiry_datetime - entry_time).total_seconds() / 60.0
    else:
        entry_day_close = entry_time.normalize() + pd.Timedelta(hours=close_h, minutes=close_m)
        minutes_left_today = min(
            minutes_per_day,
            max(0.0, (entry_day_close - entry_time).total_seconds() / 60.0),
        )

        start_next = (entry_time.normalize() + pd.Timedelta(days=1)).date()
        end_excl = expiry_datetime.normalize().date()
        full_days_between = max(
            0, int(np.busday_count(start_next, end_excl, holidays=[]))
        )

        expiry_day_open = expiry_datetime.normalize() + pd.Timedelta(hours=open_h, minutes=open_m)
        minutes_elapsed_expiry_day = min(
            minutes_per_day,
            max(0.0, (expiry_datetime - expiry_day_open).total_seconds() / 60.0),
        )

        minutes_to_expiry = (
            minutes_left_today
            + full_days_between * minutes_per_day
            + minutes_elapsed_expiry_day
        )

    return minutes_to_expiry / (minutes_per_day * trading_days_per_year)

def _d1(S: float, K: float, sigma: float, T: float) -> float:
    return (log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt(T))

def bs_price(S: float, K: float, sigma: float, T: float, right: str) -> float:
    """Black-Scholes price, r = 0."""
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K) if right == "CE" else max(0.0, K - S)
    d1 = _d1(S, K, sigma, T)
    d2 = d1 - sigma * sqrt(T)
    if right == "CE":
        return S * norm.cdf(d1) - K * norm.cdf(d2)
    return K * norm.cdf(-d2) - S * norm.cdf(-d1)

def bs_delta(S: float, K: float, sigma: float, T: float, right: str) -> float:
    d1 = _d1(S, K, sigma, T)
    return norm.cdf(d1) if right == "CE" else norm.cdf(d1) - 1.0

def bs_gamma(S: float, K: float, sigma: float, T: float) -> float:
    d1 = _d1(S, K, sigma, T)
    return norm.pdf(d1) / (S * sigma * sqrt(T))

def bs_theta_per_year(S: float, K: float, sigma: float, T: float) -> float:
    d1 = _d1(S, K, sigma, T)
    return -(S * norm.pdf(d1) * sigma) / (2 * sqrt(T))


def theta_per_bar(theta_annual: float, trading_days_per_year: int, bars_per_day: int) -> float:
    return theta_annual / trading_days_per_year / bars_per_day


def implied_vol(mkt_price: float, S: float, K: float, T: float, right: str,
                 lo: float = 1e-4, hi: float = 5.0) -> float:
    """Invert Black-Scholes for sigma via Brent's method"""
    if mkt_price is None or pd.isna(mkt_price) or mkt_price <= 0 or T <= 0:
        return float("nan")

    intrinsic = max(0.0, S - K) if right == "CE" else max(0.0, K - S)
    if mkt_price < intrinsic - 1e-9:
        return float("nan")  

    def f(sigma):
        return bs_price(S, K, sigma, T, right) - mkt_price

    try:
        f_lo, f_hi = f(lo), f(hi)
        if f_lo * f_hi > 0:
            return float("nan")
        return brentq(f, lo, hi, xtol=1e-8, maxiter=200)
    except (ValueError, RuntimeError):
        return float("nan")

def compute_implied_forward(snapshot: pd.DataFrame, spot: float, n_strikes: int = 5) -> float:
    """
    F_K = K + (C_K - P_K)   [put-call parity, r = 0]
    """
    strikes = sorted(snapshot.index.get_level_values("strike").unique())
    if not strikes:
        raise ValueError("Empty option chain snapshot")

    band = sorted(strikes, key=lambda k: abs(k - spot))[:max(1, n_strikes)]

    f_estimates = []
    for k in band:
        try:
            c = snapshot.loc[(k, "CE"), "close"]
            p = snapshot.loc[(k, "PE"), "close"]
        except KeyError:
            continue
        if pd.isna(c) or pd.isna(p):
            continue
        f_estimates.append(k + (c - p))

    if not f_estimates:
        raise ValueError("Could not find matching call/put pairs near spot to imply forward")

    return float(np.median(f_estimates))

def build_iv_curve(snapshot: pd.DataFrame, S: float, T: float) -> pd.Series:
    ivs = {}
    for (strike, right), row in snapshot.iterrows():
        ivs[(strike, right)] = implied_vol(row.get("close"), S, strike, T, right)
    idx = pd.MultiIndex.from_tuples(ivs.keys(), names=snapshot.index.names)
    return pd.Series(list(ivs.values()), index=idx, name="iv")


def build_delta_curve(iv_curve: pd.Series, S: float, T: float, right: str) -> pd.Series:
    deltas = {}
    for (strike, r), sigma in iv_curve.items():
        if r != right or pd.isna(sigma):
            continue
        deltas[strike] = bs_delta(S, strike, sigma, T, right)
    return pd.Series(deltas, name="delta").sort_index()


def select_k_star(delta_curve: pd.Series, delta_target: float, right: str) -> int:
    """
    K* = argmin_K |delta_K - delta_target|
    """
    if delta_curve.empty:
        raise ValueError("No valid per-strike deltas to select K* from")
    ref = delta_curve if right == "CE" else delta_curve.abs()
    target = abs(delta_target)
    return int((ref - target).abs().idxmin())

def taylor_reward_risk(delta: float, gamma: float, delta_F_tp: float, delta_F_sl: float):
    reward = delta * delta_F_tp + 0.5 * gamma * delta_F_tp ** 2
    risk = -(delta * delta_F_sl + 0.5 * gamma * delta_F_sl ** 2)
    return reward, risk

def compute_atr(spot_df: pd.DataFrame, period: int = 14) -> float:
    high, low, close = spot_df["high"], spot_df["low"], spot_df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return float(atr.iloc[-1]) if not atr.empty else float("nan")

def compute_hv(spot_df: pd.DataFrame, entry_time: pd.Timestamp, window_days: int = 20,
               trading_days_per_year: int = 252) -> float:
    hist = spot_df[spot_df["datetime"] <= entry_time]
    if hist.empty:
        return float("nan")
    daily_close = hist.groupby(hist["datetime"].dt.normalize())["close"].last()
    daily_close = daily_close.iloc[-(window_days + 1):]
    if len(daily_close) < 2:
        return float("nan")
    log_returns = np.log(daily_close / daily_close.shift(1)).dropna()
    if log_returns.empty:
        return float("nan")
    return float(log_returns.std() * sqrt(trading_days_per_year))

def iv_hv_filter(sigma_k_star: float, hv_20: float, threshold_ratio: float = 1.2):

    if pd.isna(sigma_k_star) or pd.isna(hv_20) or hv_20 == 0:
        return float("nan"), False
    ratio = sigma_k_star / hv_20
    return float(ratio), bool(ratio > threshold_ratio)

def orrr_filter(expected_reward: float, expected_risk: float, threshold: float = 2.0,
                 round_trip_cost: float = 0.0):
    net_reward = expected_reward - round_trip_cost
    net_risk = expected_risk + round_trip_cost
    if net_risk == 0:
        return float("inf"), False
    orrr = net_reward / net_risk
    return float(orrr), bool(orrr < threshold)

def select_k_short_target(F: float, delta_F_tp_signed: float) -> float:
    """
    K_short,target = F + delta_F_tp   (signed)
    """
    return F + (delta_F_tp_signed * 1.1)

def snap_to_nearest_strike(target: float, available_strikes) -> int:
    if not available_strikes:
        raise ValueError("No available strikes to snap the short leg to")
    return int(min(available_strikes, key=lambda k: abs(k - target)))

def select_strike(signal: dict, chain: pd.DataFrame, config: dict) -> StrikeSelection:

    required = ("delta_target", "risk_budget", "expiry_datetime")
    missing = [k for k in required if k not in config]
    if missing:
        raise ValueError(f"config missing required keys: {missing}")

    spot_df = get_spotNifty_1hr()

    entry_time = signal["timestamp"]
    direction = signal["direction"]
    S = signal["entry_price"]
    tp = signal["take_profit"]
    sl = signal["stop_loss"]

    right = "CE" if direction == "uptrend" else "PE"
    D_TP = abs(tp - S)
    D_SL = abs(S - sl)

    trading_days_per_year = 252
    bars_per_day = 375
    session_open="09:15"
    session_close="15:30"

    snapshot = get_chain_snapshot(chain, entry_time, S, prune=10)

    T = compute_T(
        entry_time,
        config["expiry_datetime"],
        session_open=session_open,
        session_close=session_close,
        trading_days_per_year=trading_days_per_year,
    )
    if T <= 0:
        raise ValueError("Non-positive time-to-expiry at entry_time; check expiry_datetime")

    F = compute_implied_forward(snapshot, S, n_strikes=5)

    iv_curve = build_iv_curve(snapshot, S, T)
    delta_curve = build_delta_curve(iv_curve, S, T, right)

    K_star = select_k_star(delta_curve, config["delta_target"], right)
    sigma_k_star = float(iv_curve[(K_star, right)])
    delta_k_star = float(delta_curve[K_star])

    gamma_k_star = bs_gamma(S, K_star, sigma_k_star, T)
    theta_k_star = theta_per_bar(
        bs_theta_per_year(S, K_star, sigma_k_star, T), trading_days_per_year, bars_per_day
    )

    delta_F_tp, delta_F_sl = (D_TP, -D_SL) if direction == "uptrend" else (-D_TP, D_SL)

    expected_reward_naked, expected_risk_naked = taylor_reward_risk(
        delta_k_star, gamma_k_star, delta_F_tp, delta_F_sl
    )
    orrr_naked = (
        expected_reward_naked / expected_risk_naked if expected_risk_naked != 0 else float("inf")
    )

    hv_20 = compute_hv(
        spot_df, entry_time,
        window_days=config.get("hv_window_days", 20),
        trading_days_per_year=trading_days_per_year,
    )
    
    iv_hv_ratio, filter1_triggered = iv_hv_filter(
        sigma_k_star, hv_20,
        threshold_ratio=config.get("iv_hv_threshold", 1.2),
    )

    premium_k_star = float(snapshot.loc[(K_star, right), "close"])
    est_premium_friction = (premium_k_star * config.get("slippage_pct", 0.025)) * 2
    
    _, filter3_triggered = orrr_filter(
        expected_reward_naked, expected_risk_naked,
        threshold=config.get("orrr_threshold", 2.0),
        round_trip_cost=est_premium_friction
    )

    route = "spread" if (filter1_triggered or filter3_triggered) else "naked_long"

    K_short = sigma_k_short = delta_k_short = gamma_k_short = theta_k_short = None
    expected_reward_spread = expected_risk_spread = orrr_spread = theta_net = None

    if route == "spread":
        available_strikes = sorted({
            s for (s, r) in snapshot.index
            if r == right and pd.notna(snapshot.loc[(s, r), "close"])
        })
        target_strike = select_k_short_target(F, delta_F_tp)
        K_short = snap_to_nearest_strike(target_strike, available_strikes)

        sigma_k_short = float(iv_curve.get((K_short, right), float("nan")))
        if pd.isna(sigma_k_short):
            short_px = snapshot.loc[(K_short, right), "close"]
            sigma_k_short = implied_vol(short_px, S, K_short, T, right)

        delta_k_short = bs_delta(S, K_short, sigma_k_short, T, right)
        gamma_k_short = bs_gamma(S, K_short, sigma_k_short, T)
        theta_k_short = theta_per_bar(
            bs_theta_per_year(S, K_short, sigma_k_short, T), trading_days_per_year, bars_per_day
        )

        reward_short, risk_short = taylor_reward_risk(
            delta_k_short, gamma_k_short, delta_F_tp, delta_F_sl
        )
        expected_reward_spread = expected_reward_naked - reward_short
        expected_risk_spread = expected_risk_naked - risk_short
        orrr_spread = (
            expected_reward_spread / expected_risk_spread
            if expected_risk_spread != 0 else float("inf")
        )
        theta_net = theta_k_star - theta_k_short

    if route == "spread":
        orrr_final = orrr_spread
        risk_per_unit = abs(expected_risk_spread)
    else:
        orrr_final = orrr_naked
        risk_per_unit = abs(expected_risk_naked)

    lot_size = config.get("lot_size", 1) 
    risk_per_contract = risk_per_unit * lot_size

    calculated_lots = int(floor(config["risk_budget"] / risk_per_contract)) if risk_per_contract > 0 else 0
    lots = min(calculated_lots, config.get("max_lots", 5))

    if route == "spread" and orrr_final < config.get("spread_orrr_threshold", 1.5):
        lots = 0

    return StrikeSelection(
        entry_time=entry_time,
        direction=direction,
        right=right,
        entry_price=S,
        take_profit=tp,
        stop_loss=sl,
        F=F,
        K_star=K_star,
        sigma_k_star=sigma_k_star,
        delta_k_star=delta_k_star,
        gamma_k_star=gamma_k_star,
        theta_k_star=theta_k_star,
        delta_F_tp=delta_F_tp,
        delta_F_sl=delta_F_sl,
        expected_reward_naked=expected_reward_naked,
        expected_risk_naked=expected_risk_naked,
        orrr_naked=orrr_naked,
        iv_hv_ratio=iv_hv_ratio,
        hv_20=hv_20,
        filter1_triggered=filter1_triggered,
        filter3_triggered=filter3_triggered,
        route=route,
        K_short=K_short,
        sigma_k_short=sigma_k_short,
        delta_k_short=delta_k_short,
        gamma_k_short=gamma_k_short,
        theta_k_short=theta_k_short,
        expected_reward_spread=expected_reward_spread,
        expected_risk_spread=expected_risk_spread,
        orrr_spread=orrr_spread,
        theta_net=theta_net,
        orrr_final=orrr_final,
        lots=lots,
    )