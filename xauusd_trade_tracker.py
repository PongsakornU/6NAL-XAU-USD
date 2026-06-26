#!/usr/bin/env python3
"""
XAUUSD Trade Tracker  ->  separate Discord channel
==================================================
Watches the trade opened by your alert bot's last signal and pings you when
something meaningful happens BEFORE the original SL/TP, so a trade never sits
in limbo with no guidance.

It auto-picks up trades: whenever xauusd_discord_signals.evaluate() produces a
fresh signal, the tracker records it and starts monitoring. On each run it
checks the open trade and alerts on:

  1. TREND FLIP  - EMA20/50 crosses against you, or price closes back through
                   the EMA200. The setup's premise is broken -> consider exit.
  2. STALL       - open longer than STALL_BARS candles without reaching the
                   first profit milestone -> dead-money, consider exit.
  3. BREAKEVEN   - price has moved +BREAKEVEN_R in your favor -> suggestion to
                   move your broker stop to entry (lock in "can't lose").
  4. TRAIL       - once in profit past TRAIL_START_R, suggests a trailing stop
                   level that follows price by TRAIL_ATR * ATR.

It also detects when price hits the original SL or TP and closes tracking.

IMPORTANT: these are SUGGESTIONS sent to you. The tracker does not place or
modify any order -- you act (or not) on your broker. Whether early exits
actually help should be backtested before you trust them; treat them as
decision aids, not rules. Not financial advice.

-------------------------------------------------------------------------
SETUP
-------------------------------------------------------------------------
    pip install requests pandas numpy
Make a SECOND Discord webhook in a separate channel (e.g. #trade-tracking)
and set TRACKER_WEBHOOK below (or env TRACKER_WEBHOOK).

Run it on the SAME schedule as your alert bot (every 15-30 min). It reads the
same market data and the same strategy logic, so signals stay consistent.
"""

import os
import sys
import json
from datetime import datetime, timezone

import requests
import pandas as pd

# Reuse the exact data + indicators + signal logic from the alert bot.
import xauusd_discord_signals as strat

# =========================== CONFIG ===========================
TRACKER_WEBHOOK = os.environ.get("TRACKER_WEBHOOK", "PASTE_YOUR_TRACKING_WEBHOOK_URL")

# Exit-signal tuning
STALL_BARS = 30             # candles open w/o hitting first milestone -> stall
                            # (30 × 4h ≈ 5 days)
STALL_PROGRESS_R = 0.5      # "milestone" = price reached +0.5R in your favor
BREAKEVEN_R = 1.0           # at +1.0R, suggest moving stop to entry
TRAIL_START_R = 1.0         # start trailing once past +1.0R
TRAIL_ATR = 2.0             # trailing distance = 2.0 * ATR behind price

TRACK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "xau_trade_tracker.json")
# ==============================================================


def load_track():
    try:
        with open(TRACK_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"open": None, "last_signal_candle": None, "alerts_sent": []}


def save_track(t):
    with open(TRACK_FILE, "w") as f:
        json.dump(t, f)


def send(text):
    headers = {"User-Agent": "XAU-Tracker/1.0 (trade tracker)"}
    r = requests.post(TRACKER_WEBHOOK, json={"username": "XAU Tracker",
                      "content": text}, headers=headers, timeout=20)
    if r.status_code not in (200, 204):
        raise RuntimeError(f"Discord error {r.status_code}: {r.text}")


def send_test(price, atr):
    """TEST_MODE: post a sample message to the tracking channel so you can confirm
    the deployed tracker can reach Discord. Does NOT create or modify any trade."""
    stall_days = STALL_BARS * 4 // 24
    text = (
        "🧪 **TEST — XAU Tracker is reachable**\n"
        f"Current price {price:.2f}, ATR {atr:.2f}.\n"
        f"When a trade is open I'll ping here on: ⚠️ trend flip, 😴 stall "
        f"(>{STALL_BARS} candles ≈ {stall_days}d with no progress), 🔒 breakeven "
        f"(+{BREAKEVEN_R:g}R), and 📈 trailing stop.\n"
        "_Test only — not a live trade alert. Turn TEST_MODE off for normal runs._"
    )
    send(text)


def progress_R(trade, price):
    """How far price has moved in your favor, in R (risk units)."""
    entry, risk = trade["entry"], trade["risk"]
    if trade["direction"] == "LONG":
        return (price - entry) / risk
    return (entry - price) / risk


def main():
    if "PASTE_YOUR" in TRACKER_WEBHOOK:
        sys.exit("Set TRACKER_WEBHOOK (separate channel) in CONFIG or env.")

    df = strat.add_indicators(strat.fetch_candles())
    c = df.iloc[-2]                      # last closed candle
    price = float(c["close"])
    atr = float(c["atr"])

    # One-time end-to-end check: set TEST_MODE=1 (or the workflow toggle) to push
    # a sample ping to the tracking channel without waiting for a live trade.
    if os.environ.get("TEST_MODE", "").lower() in ("1", "true", "yes"):
        send_test(price, atr)
        print("Test ping sent to tracking channel.")
        return

    t = load_track()

    # --- 1) pick up a NEW trade from the signal the bot ACTUALLY announced ---
    # Read it from the signal bot's state file (written this same run, by the
    # step just before this one) instead of re-evaluating. This keeps the tracker
    # in lockstep with what was sent to you: it will NOT pick up signals the bot
    # suppressed via cooldown/de-dupe, nor anything during a TEST_MODE run.
    sig = strat.load_state().get("last_signal")
    if sig and sig["candle_time"] != t.get("last_signal_candle") and t["open"] is None:
        t["open"] = {
            "direction": sig["direction"], "model": sig["model"],
            "entry": sig["entry"], "sl": sig["sl"], "tp": sig["tp"],
            "risk": abs(sig["entry"] - sig["sl"]),
            "opened_candle": sig["candle_time"], "bars_open": 0,
            "hit_milestone": False,
        }
        t["last_signal_candle"] = sig["candle_time"]
        t["alerts_sent"] = []
        save_track(t)
        print(f"Now tracking {sig['direction']} {sig['model']} @ {sig['entry']}")

    trade = t.get("open")
    if not trade:
        print("No open trade to track.")
        save_track(t)
        return

    sent = set(t.get("alerts_sent", []))
    pr = progress_R(trade, price)
    is_long = trade["direction"] == "LONG"
    arrow = "🟢LONG" if is_long else "🔴SHORT"
    tag = f"{arrow} {trade['model']} from {trade['entry']}"

    # --- terminal: original SL or TP reached -> stop tracking ---
    hit_tp = (price >= trade["tp"]) if is_long else (price <= trade["tp"])
    hit_sl = (price <= trade["sl"]) if is_long else (price >= trade["sl"])
    if hit_tp:
        send(f"🎯 **TARGET HIT** — {tag}\nPrice {price:.2f} reached TP {trade['tp']}. "
             f"Trade closed (+{trade['rr'] if 'rr' in trade else 2.0:g}R). Tracking ended.")
        t["open"] = None; save_track(t); return
    if hit_sl:
        send(f"🛑 **STOP HIT** — {tag}\nPrice {price:.2f} reached SL {trade['sl']} (-1R). "
             f"Tracking ended.")
        t["open"] = None; save_track(t); return

    # --- update bars open / milestone ---
    # Count CLOSED candles since entry (not runs), so STALL timing is correct no
    # matter how often the workflow fires. Same candle-index trick the signal
    # bot uses for its cooldown.
    times = df["datetime"].astype(str).tolist()
    cur_candle = str(c["datetime"])
    opened = trade.get("opened_candle")
    if opened in times and cur_candle in times:
        trade["bars_open"] = times.index(cur_candle) - times.index(opened)
    if pr >= STALL_PROGRESS_R:
        trade["hit_milestone"] = True

    msgs = []

    # --- 2) TREND FLIP against the position ---
    p = df.iloc[-3]
    flip = False
    if is_long:
        ema_flip = p["ema_fast"] >= p["ema_mid"] and c["ema_fast"] < c["ema_mid"]
        below_200 = c["close"] < c["ema_slow"]
        flip = ema_flip or below_200
    else:
        ema_flip = p["ema_fast"] <= p["ema_mid"] and c["ema_fast"] > c["ema_mid"]
        above_200 = c["close"] > c["ema_slow"]
        flip = ema_flip or above_200
    if flip and "flip" not in sent:
        msgs.append(("flip",
            f"⚠️ **TREND FLIP** — {tag}\nThe trend has turned against this trade "
            f"(EMA cross or price back through EMA200). The setup's premise is "
            f"broken — consider closing early. Now {pr:+.2f}R."))

    # --- 3) STALL: open too long without progress ---
    if (trade["bars_open"] >= STALL_BARS and not trade["hit_milestone"]
            and "stall" not in sent):
        msgs.append(("stall",
            f"😴 **STALLING** — {tag}\nOpen {trade['bars_open']} candles and never "
            f"reached +{STALL_PROGRESS_R:g}R. Dead money — consider closing to free "
            f"it up. Now {pr:+.2f}R."))

    # --- 4) BREAKEVEN suggestion ---
    if pr >= BREAKEVEN_R and "breakeven" not in sent:
        msgs.append(("breakeven",
            f"🔒 **MOVE STOP TO BREAKEVEN** — {tag}\nUp +{pr:.2f}R. Suggestion: move "
            f"your broker stop to entry ({trade['entry']}) so this trade can no "
            f"longer lose."))

    # --- 5) TRAILING STOP suggestion ---
    if pr >= TRAIL_START_R:
        if is_long:
            trail = round(price - TRAIL_ATR * atr, 2)
        else:
            trail = round(price + TRAIL_ATR * atr, 2)
        # only alert when the trail has moved a meaningful step (every run is noisy)
        last_trail = t.get("last_trail")
        improved = (last_trail is None or
                    (trail > last_trail if is_long else trail < last_trail))
        if improved:
            msgs.append(("trail",
                f"📈 **TRAIL STOP** — {tag}\nUp +{pr:.2f}R. Suggested trailing stop: "
                f"**{trail}** ({TRAIL_ATR:g}×ATR behind {price:.2f})."))
            t["last_trail"] = trail

    # send any new alerts (breakeven/flip/stall fire once; trail can repeat)
    for key, text in msgs:
        send(text)
        if key in ("flip", "stall", "breakeven"):
            sent.add(key)

    t["alerts_sent"] = list(sent)
    t["open"] = trade
    save_track(t)

    if not msgs:
        print(f"Tracking {trade['direction']} {trade['model']}: {pr:+.2f}R, "
              f"{trade['bars_open']} bars open. Nothing to flag.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
