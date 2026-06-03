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

# --- Indicator params ---
EMA_FAST, EMA_MID, EMA_SLOW = 20, 50, 200
RSI_PERIOD = 14
RSI_LONG_ZONE = (40, 55)        # uptrend: buy a pullback into this RSI band
RSI_SHORT_ZONE = (45, 60)       # downtrend: sell a bounce into this band
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
ATR_PERIOD = 14

# --- Risk plan (attached to every signal) ---
SL_ATR_MULT = 1.5               # stop = 1.5 * ATR from entry
RR_RATIO = 2.0                  # take-profit = 2x the risk (1:2)

# --- Anti-spam ---
COOLDOWN_BARS = 4               # min closed candles between signals
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
    macd_line = ema(df["close"], MACD_FAST) - ema(df["close"], MACD_SLOW)
    df["macd"] = macd_line
    df["macd_signal"] = ema(macd_line, MACD_SIGNAL)
    df["atr"] = atr(df, ATR_PERIOD)
    return df


# --------------------- SIGNAL LOGIC ---------------------
def evaluate(df: pd.DataFrame):
    """Use the LAST CLOSED candle (index -2; -1 may still be forming)."""
    c = df.iloc[-2]
    p = df.iloc[-3]

    macd_cross_up = (p["macd"] <= p["macd_signal"]) and (c["macd"] > c["macd_signal"])
    macd_cross_dn = (p["macd"] >= p["macd_signal"]) and (c["macd"] < c["macd_signal"])

    long_ok = (
        c["close"] > c["ema_slow"] and
        c["ema_fast"] > c["ema_mid"] > c["ema_slow"] and
        RSI_LONG_ZONE[0] <= c["rsi"] <= RSI_LONG_ZONE[1] and
        macd_cross_up
    )
    short_ok = (
        c["close"] < c["ema_slow"] and
        c["ema_fast"] < c["ema_mid"] < c["ema_slow"] and
        RSI_SHORT_ZONE[0] <= c["rsi"] <= RSI_SHORT_ZONE[1] and
        macd_cross_dn
    )

    if long_ok or short_ok:
        direction = "LONG" if long_ok else "SHORT"
        entry = c["close"]
        risk = SL_ATR_MULT * c["atr"]
        if direction == "LONG":
            sl, tp = entry - risk, entry + risk * RR_RATIO
        else:
            sl, tp = entry + risk, entry - risk * RR_RATIO
        return {
            "direction": direction,
            "candle_time": str(c["datetime"]),
            "entry": round(entry, 2),
            "sl": round(sl, 2),
            "tp": round(tp, 2),
            "rr": RR_RATIO,
            "rsi": round(c["rsi"], 1),
            "atr": round(c["atr"], 2),
        }
    return None


# --------------------- STATE ---------------------
def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"last_candle": None, "bars_since": COOLDOWN_BARS}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


# --------------------- DISCORD ---------------------
def send_discord(sig: dict):
    is_long = sig["direction"] == "LONG"
    embed = {
        "title": f"{'🟢 BUY' if is_long else '🔴 SELL'}  XAUUSD  ({INTERVAL} swing)",
        "color": 3066993 if is_long else 15158332,   # green / red
        "fields": [
            {"name": "Entry",  "value": str(sig["entry"]), "inline": True},
            {"name": "Stop",   "value": str(sig["sl"]),    "inline": True},
            {"name": f"Target (1:{sig['rr']:g})",
             "value": str(sig["tp"]), "inline": True},
            {"name": "RSI",    "value": str(sig["rsi"]),   "inline": True},
            {"name": "ATR",    "value": str(sig["atr"]),   "inline": True},
            {"name": "Candle", "value": sig["candle_time"], "inline": False},
        ],
        "footer": {"text": "Confluence: trend + EMA stack + pullback + MACD cross. "
                           "Not financial advice."},
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
        state["bars_since"] = state.get("bars_since", 0) + 1
        save_state(state)
        print(f"[{datetime.now(timezone.utc):%Y-%m-%d %H:%M}Z] No signal.")
        return

    if sig["candle_time"] == state.get("last_candle"):
        print("Already alerted this candle. Skipping.")
        return

    if state.get("bars_since", COOLDOWN_BARS) < COOLDOWN_BARS:
        print(f"In cooldown ({state['bars_since']}/{COOLDOWN_BARS} bars). Skipping.")
        state["bars_since"] += 1
        save_state(state)
        return

    send_discord(sig)
    print(f"Signal sent: {sig['direction']} @ {sig['entry']}")
    save_state({"last_candle": sig["candle_time"], "bars_since": 0})


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)