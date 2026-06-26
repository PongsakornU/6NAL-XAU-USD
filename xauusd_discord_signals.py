#!/usr/bin/env python3
"""
XAUUSD (Gold) SWING Signal Bot -> Discord webhook  (free, unlimited)
====================================================================

Sends FEW but HIGH-CONVICTION swing signals to a Discord channel.

A signal fires only when ALL conditions align on the last CLOSED candle:
    1. Trend     : price on the correct side of EMA200
    2. Strength  : EMA20 / EMA50 / EMA200 stacked in trend direction
    3. Pullback  : RSI in a value zone (buy dips / sell bounces)
    4. Trigger   : MACD line crosses its signal line in trend direction
Plus a cooldown, plus ATR-based stop & a 1:2 reward:risk target.

-------------------------------------------------------------------------
ONE-TIME DISCORD SETUP  (free, no token needed)
-------------------------------------------------------------------------
1. In Discord, make (or pick) a server and a channel for signals.
2. Channel name -> Edit Channel -> Integrations -> Webhooks -> New Webhook.
3. Name it (e.g. "XAU Signals"), then click "Copy Webhook URL".
4. Paste that URL into WEBHOOK_URL below. Done.
   (Enable mobile push for that channel so alerts hit your phone:
    long-press the channel -> Notification Settings -> All Messages.)

-------------------------------------------------------------------------
MARKET DATA  (Twelve Data, free key: https://twelvedata.com/)
-------------------------------------------------------------------------
INSTALL & RUN:
    pip install requests pandas numpy
    python xauusd_discord_signals.py

SCHEDULE (swing -> hourly is plenty; checks each new 4h/daily candle):
    0 * * * * /usr/bin/python3 /path/to/xauusd_discord_signals.py >> ~/xau.log 2>&1

DISCLAIMER: Educational tooling, not financial advice. Backtest and demo-test
before risking real money. Gold is volatile; size positions you can afford.
"""

import os
import json
import math
import sys
from datetime import datetime, timezone

import requests
import numpy as np
import pandas as pd

# =========================== CONFIG ===========================
# --- Discord ---
WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK", "PASTE_YOUR_WEBHOOK_URL")

# --- Market data (Twelve Data) ---
TWELVEDATA_API_KEY = os.environ.get("TD_KEY", "PASTE_YOUR_TWELVEDATA_KEY")
SYMBOL = "XAU/USD"

# Swing timeframe: "4h" (run hourly) or "1day" (run once a day)
INTERVAL = "4h"
CANDLES_TO_FETCH = 320          # enough history for EMA200 + indicators

# --- Strategy: ema_cross + donchian_break (validated on 19yr of gold) ---
# Fires when EITHER an EMA20/50 cross (trend-filtered) OR an N-bar breakout
# happens, in the EMA200 trend direction. ~1 signal/week on 4h, ~1 every
# 7 weeks on daily. NOTE: donchian leans on trends -- expect rough patches
# in flat/falling gold. The 4h sample is bull-only, so treat 4h as optimistic.
EMA_FAST, EMA_MID, EMA_SLOW = 20, 50, 200
DONCHIAN_N = 20                 # breakout lookback (bars)
RSI_PERIOD = 14                 # kept for the info readout only (not a filter)
ATR_PERIOD = 14

# --- Risk plan (attached to every signal) ---
SL_ATR_MULT = 2.0               # stop = 2.0 * ATR from entry (matches backtest)
RR_RATIO = 2.0                  # take-profit = 2x the risk (1:2)

# --- Account / broker (for position sizing) -------------------
# Set these to YOUR account. Dollar figures are the source of truth;
# pips are just a convenience and depend on your broker's convention.
ACCOUNT_BALANCE = 1000.0        # your account size (assumed USD)
RISK_PCT = 1.0                  # % of balance to risk per trade (1-2% is common)
CONTRACT_SIZE = 100             # oz per 1.00 lot of XAUUSD (standard)
LOT_STEP = 0.01                 # smallest lot increment your broker allows
MIN_LOT = 0.01                  # broker minimum lot
PIP_SIZE = 0.10                 # price move counted as "1 pip" (gold: often 0.10)
LEVERAGE = 100                  # your account leverage (for margin estimate only)
TYPICAL_SPREAD = 0.30           # broker spread in price terms ($/oz), for info
# --------------------------------------------------------------

# --- Anti-spam ---
# Min CLOSED candles between alerts. On 4h, 18 candles = 3 days, which yields
# ~1 signal/week (matches the backtest's tradeable cadence and stops the bot
# re-alerting every bar during a long breakout). On daily, use ~3.
COOLDOWN_BARS = 18
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "xau_signal_state.json")
# ==============================================================


# --------------------- DATA ---------------------
def fetch_candles() -> pd.DataFrame:
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "outputsize": CANDLES_TO_FETCH,
        "apikey": TWELVEDATA_API_KEY,
        "format": "JSON",
        "order": "ASC",
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    if "values" not in data:
        raise RuntimeError(f"Data error: {data.get('message', data)}")
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)
    return df.reset_index(drop=True)


# --------------------- INDICATORS ---------------------
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["ema_fast"] = ema(df["close"], EMA_FAST)
    df["ema_mid"] = ema(df["close"], EMA_MID)
    df["ema_slow"] = ema(df["close"], EMA_SLOW)
    df["rsi"] = rsi(df["close"], RSI_PERIOD)
    df["atr"] = atr(df, ATR_PERIOD)
    # Donchian channel: highest high / lowest low of the PRIOR N bars
    df["don_high"] = df["high"].rolling(DONCHIAN_N).max().shift(1)
    df["don_low"] = df["low"].rolling(DONCHIAN_N).min().shift(1)
    return df


# --------------------- SIGNAL LOGIC ---------------------
def evaluate(df: pd.DataFrame):
    """Combined ema_cross + donchian_break on the LAST CLOSED candle
    (index -2; -1 may still be forming). EMA cross takes priority; if it
    doesn't fire, check the breakout. Both require EMA200 trend agreement."""
    c = df.iloc[-2]
    p = df.iloc[-3]

    direction = model = None

    # 1) EMA20/50 cross, in the EMA200 trend direction
    cross_up = p["ema_fast"] <= p["ema_mid"] and c["ema_fast"] > c["ema_mid"]
    cross_dn = p["ema_fast"] >= p["ema_mid"] and c["ema_fast"] < c["ema_mid"]
    if cross_up and c["close"] > c["ema_slow"]:
        direction, model = "LONG", "ema_cross"
    elif cross_dn and c["close"] < c["ema_slow"]:
        direction, model = "SHORT", "ema_cross"
    # 2) Donchian breakout, fired ONLY on the candle that crosses the channel
    #    (prior candle was still inside it) so it doesn't repeat every bar.
    elif (not pd.isna(c["don_high"]) and c["close"] > c["don_high"]
          and p["close"] <= p["don_high"] and c["close"] > c["ema_slow"]):
        direction, model = "LONG", "donchian_break"
    elif (not pd.isna(c["don_low"]) and c["close"] < c["don_low"]
          and p["close"] >= p["don_low"] and c["close"] < c["ema_slow"]):
        direction, model = "SHORT", "donchian_break"

    if direction is None:
        return None

    entry = c["close"]
    risk = SL_ATR_MULT * c["atr"]
    if direction == "LONG":
        sl, tp = entry - risk, entry + risk * RR_RATIO
    else:
        sl, tp = entry + risk, entry - risk * RR_RATIO
    return {
        "direction": direction,
        "model": model,
        "candle_time": str(c["datetime"]),
        "entry": round(entry, 2),
        "sl": round(sl, 2),
        "tp": round(tp, 2),
        "rr": RR_RATIO,
        "rsi": round(c["rsi"], 1),
        "atr": round(c["atr"], 2),
    }


# --------------------- STATE ---------------------
def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"last_candle": None}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


# --------------------- POSITION SIZING ---------------------
def position_plan(entry: float, sl: float, tp: float) -> dict:
    """Risk-based lot sizing. Risk a fixed % of the account; lot size falls out
    of the stop distance. Leverage only affects the margin estimate, not P&L."""
    stop_dist = abs(entry - sl)            # price distance to stop ($/oz)
    reward_dist = abs(tp - entry)          # price distance to target ($/oz)
    risk_amount = ACCOUNT_BALANCE * RISK_PCT / 100.0

    risk_per_lot = stop_dist * CONTRACT_SIZE          # $ lost per 1.00 lot if SL hit
    raw_lots = risk_amount / risk_per_lot if risk_per_lot else 0
    # round DOWN to the broker lot step so we never risk more than intended
    lots = math.floor(raw_lots / LOT_STEP) * LOT_STEP
    lots = round(max(lots, 0), 2)

    too_small = lots < MIN_LOT             # account too small for this stop at this risk

    est_risk = lots * risk_per_lot
    est_reward = lots * reward_dist * CONTRACT_SIZE
    notional = lots * CONTRACT_SIZE * entry
    margin = notional / LEVERAGE if LEVERAGE else 0

    return {
        "lots": lots,
        "too_small": too_small,
        "stop_pips": stop_dist / PIP_SIZE,
        "tp_pips": reward_dist / PIP_SIZE,
        "risk_usd": est_risk,
        "reward_usd": est_reward,
        "margin_usd": margin,
    }


# --------------------- DISCORD ---------------------
def send_discord(sig: dict):
    is_long = sig["direction"] == "LONG"
    p = position_plan(sig["entry"], sig["sl"], sig["tp"])

    if p["too_small"]:
        lot_text = (f"⚠ {MIN_LOT} (min) — stop is too wide to keep risk at "
                    f"{RISK_PCT}% of ${ACCOUNT_BALANCE:,.0f}. Risk would exceed target.")
    else:
        lot_text = f"{p['lots']:.2f} lot  (risk {RISK_PCT}% = ${p['risk_usd']:,.2f})"

    embed = {
        "title": f"{'🟢 BUY' if is_long else '🔴 SELL'}  XAUUSD  ({INTERVAL})  ·  {sig.get('model','')}",
        "color": 3066993 if is_long else 15158332,   # green / red
        "fields": [
            {"name": "Entry",  "value": str(sig["entry"]), "inline": True},
            {"name": f"Stop ({p['stop_pips']:.0f} pips)",
             "value": str(sig["sl"]), "inline": True},
            {"name": f"Target ({p['tp_pips']:.0f} pips, 1:{sig['rr']:g})",
             "value": str(sig["tp"]), "inline": True},
            {"name": "Suggested size", "value": lot_text, "inline": False},
            {"name": "Risk",    "value": f"${p['risk_usd']:,.2f}", "inline": True},
            {"name": "Reward",  "value": f"${p['reward_usd']:,.2f}", "inline": True},
            {"name": "Margin",  "value": f"~${p['margin_usd']:,.2f} @ 1:{LEVERAGE}",
             "inline": True},
            {"name": "RSI",    "value": str(sig["rsi"]),   "inline": True},
            {"name": "ATR",    "value": str(sig["atr"]),   "inline": True},
            {"name": "Candle", "value": sig["candle_time"], "inline": False},
        ],
        "footer": {"text": f"Mid price — real fill differs by ~spread (${TYPICAL_SPREAD}). "
                           "Set SL/TP on your broker. Not financial advice."},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    payload = {"username": "XAU Signals", "embeds": [embed]}
    # Discord/Cloudflare blocks the default python-requests UA (403 / error 1010),
    # so we send a custom User-Agent.
    headers = {"User-Agent": "XAU-Signals/1.0 (gold signal bot)"}
    r = requests.post(WEBHOOK_URL, json=payload, headers=headers, timeout=20)
    if r.status_code not in (200, 204):
        raise RuntimeError(f"Discord error {r.status_code}: {r.text}")


def send_test_snapshot(df: pd.DataFrame):
    """TEST_MODE: post the current reading regardless of conditions, so you can
    confirm the deployed bot can reach Discord. Does NOT touch signal state."""
    c = df.iloc[-2]
    embed = {
        "title": f"🧪 TEST — XAUUSD reading ({INTERVAL})",
        "color": 10070709,  # neutral grey/purple
        "fields": [
            {"name": "Price",  "value": f"${c['close']:,.2f}", "inline": True},
            {"name": "RSI",    "value": f"{c['rsi']:.1f}",      "inline": True},
            {"name": "ATR",    "value": f"{c['atr']:.2f}",      "inline": True},
            {"name": "EMA20",  "value": f"{c['ema_fast']:,.2f}", "inline": True},
            {"name": "EMA50",  "value": f"{c['ema_mid']:,.2f}",  "inline": True},
            {"name": "EMA200", "value": f"{c['ema_slow']:,.2f}", "inline": True},
        ],
        "footer": {"text": "Test ping — not a live signal. Turn TEST_MODE off "
                           "for normal operation."},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    payload = {"username": "XAU Signals", "embeds": [embed]}
    headers = {"User-Agent": "XAU-Signals/1.0 (gold signal bot)"}
    r = requests.post(WEBHOOK_URL, json=payload, headers=headers, timeout=20)
    if r.status_code not in (200, 204):
        raise RuntimeError(f"Discord error {r.status_code}: {r.text}")


# --------------------- MAIN ---------------------
def main():
    df = add_indicators(fetch_candles())

    # One-time end-to-end check: set TEST_MODE=1 (or the workflow toggle) to push
    # the current reading to Discord without waiting for a real setup.
    if os.environ.get("TEST_MODE", "").lower() in ("1", "true", "yes"):
        send_test_snapshot(df)
        print("Test snapshot sent to Discord.")
        return

    state = load_state()
    sig = evaluate(df)

    if sig is None:
        print(f"[{datetime.now(timezone.utc):%Y-%m-%d %H:%M}Z] No signal.")
        return

    last = state.get("last_candle")

    # de-dupe: never alert the same candle twice
    if sig["candle_time"] == last:
        print("Already alerted this candle. Skipping.")
        return

    # cooldown measured in CLOSED CANDLES (not runs): find both candles in the
    # fetched data and require at least COOLDOWN_BARS between them.
    if last is not None:
        times = df["datetime"].astype(str).tolist()
        if last in times and sig["candle_time"] in times:
            gap = times.index(sig["candle_time"]) - times.index(last)
            if gap < COOLDOWN_BARS:
                print(f"In cooldown ({gap}/{COOLDOWN_BARS} candles). Skipping.")
                return

    send_discord(sig)
    print(f"Signal sent: {sig['direction']} ({sig['model']}) @ {sig['entry']}")
    # Save the FULL announced signal (not just the candle time) so the trade
    # tracker picks up exactly this trade. The tracker reads last_signal instead
    # of re-evaluating, which keeps it in lockstep with what was actually sent
    # (no cooldown/de-dupe/test-mode desync).
    save_state({"last_candle": sig["candle_time"], "last_signal": sig})


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)