"""
AI TRADER V3 - GOLD (XAUUSD) M15, ICT/SMC-style, multi-LLM council
via Experiential Labs + MetaTrader5.

This build adds the fixes needed before pointing it at a real
account: tight-SL / risk-overrun protection, order_check validation,
a portfolio-level risk cap, a daily loss circuit breaker, and a
JSONL trade journal (decisions + closed trades).

Safety is still controlled by DEMO_ONLY / DRY_RUN in your .env.
This script does not flip those for you - going live is your
deliberate decision, made after a real forward-test track record.
"""

import os
import io
import re
import json
import math
import time
import base64
import logging
import datetime
from concurrent.futures import ThreadPoolExecutor

import MetaTrader5 as mt5
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import mplfinance as mpf
from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

API_KEY = os.getenv("EXPLABS_API_KEY")
if not API_KEY:
    raise RuntimeError("EXPLABS_API_KEY not found in .env")

BASE_URL = "https://api.experientiallabs.ai/v1"

SYMBOL = os.getenv("MT5_SYMBOL", "GOLD")

TIMEFRAME = mt5.TIMEFRAME_M15
TIMEFRAME_NAME = "M15"

BARS = int(os.getenv("MT5_BARS", "200"))

# --- Safety switches -----------------------------------------
DEMO_ONLY = os.getenv("DEMO_ONLY", "true").lower() == "true"
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"

# --- Per-trade risk --------------------------------------------
# Sane default if you forget to set it. 10% (the old default) is
# not a "per trade" number any real prop desk would use.
RISK_PERCENT = float(os.getenv("RISK_PER_TRADE", "1.0"))

MAX_OPEN_TRADES = int(os.getenv("MAX_OPEN_TRADES", "3"))

MIN_CONFIDENCE = float(os.getenv("AI_MIN_CONFIDENCE", "55"))

# In a HIGH-volatility regime (see detect_regime), require a
# materially higher confidence bar before trading - expansion/stop-
# hunt conditions punish marginal setups harder than calm ones.
MIN_CONFIDENCE_HIGH_VOL = float(os.getenv("AI_MIN_CONFIDENCE_HIGH_VOL", "70"))

# SL must be at least this many ATRs away from entry. Rejects
# unrealistically tight stops the AI can hallucinate, which
# would otherwise blow up position size (see calculate_volume).
MIN_SL_ATR_MULT = float(os.getenv("MIN_SL_ATR_MULT", "0.5"))

# If the broker's volume_min forces the actual $ risk above the
# intended RISK_PERCENT by more than this multiple, reject the
# trade instead of silently taking oversized risk.
MAX_RISK_OVERRUN_MULT = float(os.getenv("MAX_RISK_OVERRUN_MULT", "1.3"))

# --- Portfolio-level caps ---------------------------------------
# Total risk allowed across ALL open positions on this symbol at
# once (they are not diversified - it's the same instrument).
MAX_TOTAL_RISK_PERCENT = float(os.getenv("MAX_TOTAL_RISK_PERCENT", "3.0"))

# Daily circuit breaker: if realized balance drawdown since the
# start of the trading day hits this, stop opening new trades
# until the next day.
MAX_DAILY_LOSS_PERCENT = float(os.getenv("MAX_DAILY_LOSS_PERCENT", "3.0"))

LOOP_SECONDS = int(os.getenv("LOOP_SECONDS", "5"))

MAGIC = int(os.getenv("MAGIC", "990011"))
DEVIATION = int(os.getenv("DEVIATION", "30"))

ENTRY_TOLERANCE_ATR = float(os.getenv("ENTRY_TOLERANCE_ATR", "0.20"))
SETUP_MAX_SECONDS = 15 * 60

CHART_BARS = int(os.getenv("CHART_BARS", "80"))
USE_CHART_VISION = os.getenv("USE_CHART_VISION", "true").lower() == "true"

JOURNAL_PATH = os.getenv("JOURNAL_PATH", "logs/trade_journal.jsonl")


# ============================================================
# LOGGING
# ============================================================

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler("logs/trader.log", encoding="utf-8")]
)

log = logging.getLogger("AI_TRADER")


def journal_write(record):
    """Append one JSON line to the trade journal. Never raises."""

    record = dict(record)
    record["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")

    try:
        with open(JOURNAL_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        log.error("Journal write failed: %s", e)


# ============================================================
# AI CLIENT
# ============================================================

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


class RiskRejected(Exception):
    """Raised when a setup fails a risk sanity check (not a bug -
    a deliberate refusal to trade it)."""


# ============================================================
# GLOBAL STATE
# ============================================================

pending_setup = None
last_analysis_bar = None
last_ai_decision = None
last_market_state = None
last_analysis_time = None
last_order_status = "SYSTEM STARTING"
last_team_results = []
last_error = ""

seen_deal_tickets = set()

day_start_date = None
day_start_balance = None
trading_halted = False
halt_reason = ""


# ============================================================
# TERMINAL SCREEN
# ============================================================

def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def print_screen():
    global pending_setup, last_ai_decision, last_analysis_time
    global last_order_status, last_error

    clear_screen()
    now = time.strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 66)
    print("                    AI TRADER V3")
    print("             EXPERIENTIAL LABS + MT5")
    print("=" * 66)
    print(f"TIMEFRAME     : {SYMBOL} {TIMEFRAME_NAME}")
    print(f"TIME          : {now}")
    print("-" * 66)

    account = mt5.account_info()
    if account:
        print(f"BALANCE       : ${account.balance:.2f}")
        print(f"EQUITY        : ${account.equity:.2f}")
    else:
        print("BALANCE       : unavailable")
        print("EQUITY        : unavailable")

    tick = mt5.symbol_info_tick(SYMBOL)
    if tick:
        bid = float(tick.bid)
        ask = float(tick.ask)
        print(f"BID           : {bid:.2f}")
        print(f"ASK           : {ask:.2f}")
    else:
        bid = 0
        ask = 0
        print("MARKET        : unavailable")

    print("-" * 66)

    positions = get_open_positions()
    open_risk = calculate_open_risk_percent()

    print(f"OPEN TRADES   : {len(positions)} / {MAX_OPEN_TRADES}")
    print(f"OPEN RISK     : {open_risk:.2f}% / {MAX_TOTAL_RISK_PERCENT:.2f}% cap")
    print(f"RISK / TRADE  : {RISK_PERCENT:.2f}%")
    print(f"DRY RUN       : {DRY_RUN}")
    print(f"DEMO ONLY     : {DEMO_ONLY}")

    if day_start_balance and account:
        daily_pct = (account.balance - day_start_balance) / day_start_balance * 100
        print(f"DAILY P/L     : {daily_pct:+.2f}% (limit -{MAX_DAILY_LOSS_PERCENT:.2f}%)")

    if trading_halted:
        print(f"TRADING       : HALTED - {halt_reason}")

    print("-" * 66)
    print("                  MARKET REGIME")
    print("-" * 66)

    if last_market_state and "regime" in last_market_state:
        r = last_market_state["regime"]
        htf = last_market_state.get("htf_context") or {}
        print(f"TREND STRENGTH: {r['trend_strength']} (ADX {r['adx']})")
        print(f"VOLATILITY    : {r['volatility']} (ATR pct {r['atr_percentile']})")
        print(f"H1 / H4 BIAS  : {htf.get('h1_trend', '?')} / {htf.get('h4_trend', '?')}")
        if htf.get("prev_day_high") and htf.get("prev_day_low"):
            print(f"PDH / PDL     : {htf['prev_day_high']:.2f} / {htf['prev_day_low']:.2f}")
    else:
        print("No regime data yet.")

    print("-" * 66)
    print("                    AI DECISION")
    print("-" * 66)

    if last_ai_decision:
        d = last_ai_decision
        print(f"ACTION        : {d['action']}")
        print(f"CONFIDENCE    : {d['confidence']:.1f}%")
        print(f"ENTRY         : {d['entry']:.2f}")
        print(f"SL            : {d['sl']:.2f}")
        print(f"TP            : {d['tp']:.2f}")
        if d["reason"]:
            print(f"REASON        : {d['reason'][:80]}")
    else:
        print("ACTION        : WAIT")
        print("No AI decision yet.")

    print("-" * 66)
    print("                  ENTRY MONITOR")
    print("-" * 66)

    if pending_setup:
        p = pending_setup
        entry = p["entry"]
        current = ask if p["action"] == "BUY" else bid
        distance = abs(current - entry)
        tolerance = max(p["atr"] * ENTRY_TOLERANCE_ATR, 0.10)
        age = int(time.time() - p["created"])

        print(f"SETUP         : {p['action']}")
        print(f"ENTRY         : {entry:.2f}")
        print(f"CURRENT       : {current:.2f}")
        print(f"DISTANCE      : {distance:.2f}")
        print(f"TOLERANCE     : {tolerance:.2f}")
        print(f"SETUP AGE     : {age}s")
        print()

        if distance <= tolerance:
            print("STATUS        : ENTRY REACHED")
        else:
            print("STATUS        : WAITING FOR ENTRY")
    else:
        print("SETUP         : NONE")
        print(f"STATUS        : {last_order_status}")

    print("-" * 66)

    if last_analysis_time:
        print(f"LAST ANALYSIS : {last_analysis_time}")

    if last_error:
        print("-" * 66)
        print(f"LAST ERROR    : {last_error[:100]}")

    print("=" * 66)
    print(f"Next screen update in {LOOP_SECONDS} seconds...")


# ============================================================
# JSON PARSER
# ============================================================

def parse_ai(text):
    text = text.strip()
    text = re.sub(r"```json", "", text, flags=re.I)
    text = re.sub(r"```", "", text)

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("AI did not return JSON")

    data = json.loads(match.group(0))

    action = str(data.get("action", "WAIT")).upper()
    if action not in ["BUY", "SELL", "WAIT"]:
        action = "WAIT"

    return {
        "action": action,
        "confidence": float(data.get("confidence", 0)),
        "entry": float(data.get("entry", 0)),
        "sl": float(data.get("sl", 0)),
        "tp": float(data.get("tp", 0)),
        "reason": str(data.get("reason", "")),
    }


# ============================================================
# FIND WORKING MODELS + PROBE VISION SUPPORT
# ============================================================

def find_models():
    log.info("Loading Experiential Labs models...")

    models = client.models.list()
    ids = [m.id for m in models.data if getattr(m, "id", None)]

    priority = [
        "claude-opus", "claude-sonnet", "claude",
        "qwen", "deepseek", "gpt", "gemini"
    ]

    ids.sort(
        key=lambda x: next(
            (len(priority) - i for i, name in enumerate(priority)
             if name in x.lower()),
            0
        ),
        reverse=True
    )

    working = []

    for model in ids:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Reply only OK."}],
                max_tokens=5
            )
            if response.choices:
                working.append(model)
                log.info("AI MODEL OK: %s", model)
        except Exception:
            log.info("Model skipped: %s", model)

        if len(working) >= 3:
            break

    if not working:
        raise RuntimeError("No working AI model found.")

    while len(working) < 3:
        working.append(working[0])

    return working


# A 1x1 transparent PNG, used only to probe whether a model
# actually accepts image input without wasting a real chart call.
_PROBE_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
    "z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def probe_vision_support(model):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Reply only OK."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{_PROBE_PNG_B64}"
                        }
                    }
                ]
            }],
            max_tokens=5
        )
        return bool(response.choices)
    except Exception:
        return False


AI_MODELS = find_models()

STRUCTURE_MODEL = AI_MODELS[0]
LIQUIDITY_MODEL = AI_MODELS[1]
QUANT_MODEL = AI_MODELS[2]
JUDGE_MODEL = AI_MODELS[0]

VISION_CAPABLE = {}
for _m in set(AI_MODELS):
    VISION_CAPABLE[_m] = probe_vision_support(_m) if USE_CHART_VISION else False
    log.info("Vision support - %s: %s", _m, VISION_CAPABLE[_m])

if USE_CHART_VISION and not any(VISION_CAPABLE.values()):
    log.warning(
        "USE_CHART_VISION=true but none of the selected models "
        "accepted an image in the probe - every call will fall "
        "back to text-only analysis."
    )


# ============================================================
# MT5
# ============================================================

def connect_mt5():
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

    account = mt5.account_info()
    if account is None:
        raise RuntimeError("Cannot read MT5 account")

    if DEMO_ONLY and account.trade_mode == mt5.ACCOUNT_TRADE_MODE_REAL:
        raise RuntimeError("REAL ACCOUNT DETECTED. TRADING BLOCKED.")

    if not mt5.symbol_select(SYMBOL, True):
        raise RuntimeError(f"Cannot select {SYMBOL}")


# ============================================================
# MARKET DATA
# ============================================================

def get_market():
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, BARS)

    if rates is None or len(rates) < 60:
        raise RuntimeError("Not enough market data")

    df = pd.DataFrame(rates)

    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()

    previous = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - previous).abs()
    tr3 = (df["low"] - previous).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    df["atr"] = tr.rolling(14).mean()
    df["momentum"] = df["close"].pct_change(5) * 100

    # --- Trend strength (Wilder's ADX, period 14) -----------------
    # Used to tell BUY/SELL-style "continuation" conditions (high
    # ADX = trending) apart from "sweep and reverse" conditions
    # (low ADX = ranging), instead of guessing this from the chart.
    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move

    atr_wilder = tr.ewm(alpha=1 / 14, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr_wilder)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr_wilder)

    di_sum = (plus_di + minus_di).replace(0, float("nan"))
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    df["adx"] = dx.ewm(alpha=1 / 14, adjust=False).mean().fillna(0)
    df["plus_di"] = plus_di.fillna(0)
    df["minus_di"] = minus_di.fillna(0)

    # --- Volatility regime (ATR percentile vs recent history) -----
    # Where today's ATR sits vs the last 100 bars - tells us if we
    # are in an unusually quiet or unusually explosive market,
    # independent of trend direction.
    df["atr_percentile"] = (
        df["atr"].rolling(100, min_periods=20)
        .apply(lambda w: (w.iloc[-1] > w).mean() * 100, raw=False)
    )

    return df


def detect_regime(df):
    """Classifies the CURRENT bar's conditions using indicators the
    code already computed - not left to the model to eyeball off
    the chart. Returns a dict merged straight into market_state."""

    last = df.iloc[-1]
    adx = float(last["adx"])
    atr_pctile = float(last["atr_percentile"]) if pd.notna(last["atr_percentile"]) else 50.0

    # ADX thresholds are the standard Wilder convention:
    # <15 no clear trend, 15-25 developing, >25 trending, >40 strong.
    if adx >= 25:
        trend_strength = "TRENDING"
    elif adx <= 15:
        trend_strength = "RANGING"
    else:
        trend_strength = "TRANSITIONAL"

    if atr_pctile >= 80:
        volatility = "HIGH"
    elif atr_pctile <= 20:
        volatility = "LOW"
    else:
        volatility = "NORMAL"

    return {
        "trend_strength": trend_strength,
        "adx": round(adx, 1),
        "volatility": volatility,
        "atr_percentile": round(atr_pctile, 1),
    }


# H1/H4 are only used for directional bias/context, not for entries,
# so a lightweight EMA20 vs EMA50 read is enough - no need to
# duplicate the full M15 indicator stack.
def _htf_trend(symbol, timeframe, bars=120):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None or len(rates) < 60:
        return "UNKNOWN", None, None

    df = pd.DataFrame(rates)
    ema20 = df["close"].ewm(span=20, adjust=False).mean()
    ema50 = df["close"].ewm(span=50, adjust=False).mean()

    price = float(df["close"].iloc[-1])
    e20 = float(ema20.iloc[-1])
    e50 = float(ema50.iloc[-1])

    if price > e20 > e50:
        trend = "BULLISH"
    elif price < e20 < e50:
        trend = "BEARISH"
    else:
        trend = "NEUTRAL"

    return trend, float(df["high"].iloc[-1]), float(df["low"].iloc[-1])


def get_htf_context():
    """Higher-timeframe bias (H1 + H4) plus previous day's
    high/low - gives the model the top-down context a discretionary
    trader would check before trusting an M15 signal, instead of
    reading M15 in isolation."""

    h1_trend, _, _ = _htf_trend(SYMBOL, mt5.TIMEFRAME_H1, bars=120)
    h4_trend, _, _ = _htf_trend(SYMBOL, mt5.TIMEFRAME_H4, bars=120)

    daily = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_D1, 1, 1)  # yesterday's completed day
    if daily is not None and len(daily) == 1:
        prev_day_high = round(float(daily[0]["high"]), 2)
        prev_day_low = round(float(daily[0]["low"]), 2)
    else:
        prev_day_high = None
        prev_day_low = None

    return {
        "h1_trend": h1_trend,
        "h4_trend": h4_trend,
        "prev_day_high": prev_day_high,
        "prev_day_low": prev_day_low,
    }


def build_market_state(df, htf=None):
    last = df.iloc[-1]

    price = float(last["close"])
    ema20 = float(last["ema20"])
    ema50 = float(last["ema50"])
    atr = float(last["atr"])
    momentum = float(last["momentum"])

    if price > ema20 and ema20 > ema50:
        trend = "BULLISH"
    elif price < ema20 and ema20 < ema50:
        trend = "BEARISH"
    else:
        trend = "NEUTRAL"

    if momentum > 0.05:
        momentum_state = "BULLISH"
    elif momentum < -0.05:
        momentum_state = "BEARISH"
    else:
        momentum_state = "NEUTRAL"

    candles = []
    for _, row in df.iloc[-60:].iterrows():
        candles.append({
            "time": str(row["time"]),
            "open": round(float(row["open"]), 2),
            "high": round(float(row["high"]), 2),
            "low": round(float(row["low"]), 2),
            "close": round(float(row["close"]), 2),
            "volume": int(row["tick_volume"]),
        })

    state = {
        "symbol": SYMBOL,
        "timeframe": TIMEFRAME_NAME,
        "price": round(price, 2),
        "trend": trend,
        "momentum": momentum_state,
        "momentum_value": round(momentum, 4),
        "ema20": round(ema20, 2),
        "ema50": round(ema50, 2),
        "atr": round(atr, 2),
        "recent_high": round(float(df.iloc[-20:]["high"].max()), 2),
        "recent_low": round(float(df.iloc[-20:]["low"].min()), 2),
        "candles": candles,
    }

    state["regime"] = detect_regime(df)

    if htf:
        state["htf_context"] = htf

    return state


# ============================================================
# CHART RENDER (for vision models)
# ============================================================

def render_chart_image(df, n=CHART_BARS):
    plot_df = df.iloc[-n:].copy()
    plot_df["time"] = pd.to_datetime(plot_df["time"], unit="s")
    plot_df.set_index("time", inplace=True)

    plot_df = plot_df.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "tick_volume": "Volume",
    })

    apds = [
        mpf.make_addplot(plot_df["ema20"], color="dodgerblue", width=1.1),
        mpf.make_addplot(plot_df["ema50"], color="orange", width=1.1),
    ]

    buf = io.BytesIO()
    mpf.plot(
        plot_df, type="candle", style="charles", addplot=apds,
        volume=False, figsize=(11, 6),
        savefig=dict(fname=buf, format="png", dpi=120, bbox_inches="tight"),
    )
    buf.seek(0)

    return base64.b64encode(buf.read()).decode("utf-8")


def build_vision_content(prompt, chart_b64):
    content = [{"type": "text", "text": prompt}]

    if chart_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{chart_b64}"}
        })

    return content


# ============================================================
# AI SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a professional XAUUSD discretionary trader using
ICT / Smart Money Concepts. You trade the M15 chart during
the New York session, the same way an experienced ICT trader
would - by actually reading the chart, not just the numbers.

You will usually be given a chart IMAGE (candles + EMA20/EMA50)
together with structured market data. The market data includes a
"regime" block (trend_strength: TRENDING/RANGING/TRANSITIONAL,
adx, volatility: HIGH/NORMAL/LOW, atr_percentile) computed
directly from the price series, and an "htf_context" block
(h1_trend, h4_trend, prev_day_high/low). These are FACTS, not
suggestions - use them to decide HOW to read the chart, the way a
discretionary trader changes approach depending on conditions
instead of applying one template blindly:

- If regime.trend_strength is TRENDING: look for CONTINUATION -
  pullback entries into a discount (in an uptrend) or premium (in
  a downtrend) of the current impulse leg, in the direction of the
  M15 structure. Only take counter-trend setups against a TRENDING
  regime if there is a clean, confirmed CHoCH - don't fade a strong
  trend on a marginal signal.
- If regime.trend_strength is RANGING: look for LIQUIDITY SWEEP +
  REVERSAL at the edges of the recent range (equal highs/lows,
  PDH/PDL) rather than continuation - trending-style breakout
  entries are unreliable in a ranging market and should be treated
  with more suspicion.
- If regime.volatility is HIGH: either require a materially cleaner
  setup and wider SL (beyond the ATR minimum the system already
  enforces), or prefer WAIT - expansion candles and stop hunts are
  more likely to invalidate a normal-sized stop.
- If regime.volatility is LOW: be cautious of setups that look
  clean only because the market isn't moving - a breakout of a
  low-volatility range can be a genuine expansion or a trap, so
  weight structure/liquidity evidence over the chart's calm look.
- Use htf_context to filter direction, not to pick entries: avoid
  taking M15 setups that fight both h1_trend AND h4_trend unless
  there is a strong, clearly confirmed reversal structure (sweep +
  CHoCH) - a counter-HTF-trend scalp needs materially more evidence
  than a with-HTF-trend continuation.
- Determine current M15 structure: BOS / CHoCH / MSS, and whether
  price is trading premium or discount relative to the last
  swing range.
- Identify liquidity: equal highs/lows, PDH/PDL (from htf_context),
  obvious stop clusters above/below recent swing points, and
  whether a liquidity sweep already happened or is still likely.
- Identify unmitigated FVG / iFVG and Order Blocks on the
  visible range.
- Consider session context: this is GOLD M15 in/around the NY
  killzone - factor in typical NY volatility and judas-swing
  behavior.
- Prefer entries inside an OTE zone (61.8-79% retracement) of
  the most recent impulse leg, not chasing the current price.

You do NOT know the future with certainty. You have a
probabilistic edge based on structure, liquidity, and regime
context - not a guarantee. Size your confidence accordingly - a
clean, confirmed setup that aligns with the current regime and HTF
bias deserves higher confidence than a marginal or counter-regime
one.

Your SL must sit beyond the real invalidation point of your
setup (past the order block / swing point that disproves the
idea), not an arbitrary tight number - a stop that's too close
to entry will be rejected by the trading system regardless of
your confidence.

Possible actions: BUY, SELL, WAIT.

WAIT is completely valid. NEVER force a trade. If there is no
clean ICT setup, choose WAIT.

If BUY or SELL is selected, you MUST provide entry, sl, tp.

ENTRY is extremely important. The trading system will WAIT for
price to reach your ENTRY before executing - so choose the zone
you actually want to be filled at (e.g. an OTE / order block /
FVG edge), not the current market price. Do not chase price.

Return ONLY JSON:

{
 "action": "BUY|SELL|WAIT",
 "confidence": 0-100,
 "entry": 0,
 "sl": 0,
 "tp": 0,
 "reason": "short explanation naming the ICT element(s) used"
}
"""


# ============================================================
# AI ANALYST / JUDGE CALLS
# ============================================================

def ask_ai(model, role, state, chart_b64=None):
    prompt = f"""
You are the {role} specialist on the trading team.

Analyze GOLD on M15 - look at the attached chart image first (if
present), then cross-check with the structured data below.

MARKET DATA:

{json.dumps(state, indent=2)}

ROLE FOCUS:

{role}

Decide BUY, SELL, or WAIT. Do not force a trade.

If trading, determine ENTRY, SL, TP as described in the system
instructions.

Return only JSON.
"""

    use_vision = bool(
        USE_CHART_VISION and chart_b64 and VISION_CAPABLE.get(model, False)
    )

    def call(vision):
        content = build_vision_content(prompt, chart_b64) if vision else prompt
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ]
        )

    try:
        response = call(use_vision)
    except Exception as e:
        log.info("Vision call failed for %s (%s), retrying text-only", model, e)
        response = call(False)

    return parse_ai(response.choices[0].message.content)


def run_team(state, chart_b64=None):
    jobs = [
        (STRUCTURE_MODEL, "MARKET STRUCTURE"),
        (LIQUIDITY_MODEL, "LIQUIDITY"),
        (QUANT_MODEL, "QUANTITATIVE ANALYSIS"),
    ]

    results = []

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(ask_ai, model, role, state, chart_b64)
            for model, role in jobs
        ]

        for future in futures:
            try:
                results.append(future.result())
            except Exception as e:
                results.append({
                    "action": "WAIT", "confidence": 0,
                    "entry": 0, "sl": 0, "tp": 0, "reason": str(e),
                })

    return results


def run_judge(state, analyses, chart_b64=None):
    prompt = f"""
You are the FINAL autonomous trading decision maker.

Look at the attached chart image yourself first (if present) -
form your own read of structure/liquidity/FVGs/order blocks -
then use the team's analyses below as extra input, not as a vote
you must follow blindly. You can override the team if the chart
clearly disagrees with them.

You receive:
1. GOLD M15 market data
2. Market Structure AI
3. Liquidity AI
4. Quant AI

MARKET:

{json.dumps(state, indent=2)}

AI TEAM:

{json.dumps(analyses, indent=2)}

Make the FINAL decision: BUY, SELL, or WAIT. Do not force a trade.

If BUY or SELL: choose a realistic ENTRY (an OTE / order block /
FVG edge you are actually waiting for - not the current price).
The system will WAIT for the ENTRY. Choose SL beyond the real
invalidation point of your setup, and TP based on market
structure. Do not chase price.

Return ONLY JSON.
"""

    use_vision = bool(
        USE_CHART_VISION and chart_b64 and VISION_CAPABLE.get(JUDGE_MODEL, False)
    )

    def call(vision):
        content = build_vision_content(prompt, chart_b64) if vision else prompt
        return client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ]
        )

    try:
        response = call(use_vision)
    except Exception as e:
        log.info("Vision call failed for judge (%s), retrying text-only", e)
        response = call(False)

    return parse_ai(response.choices[0].message.content)


# ============================================================
# OPEN POSITIONS / PORTFOLIO RISK
# ============================================================

def get_open_positions():
    positions = mt5.positions_get(symbol=SYMBOL)
    return list(positions) if positions is not None else []


def calculate_open_risk_percent():
    """Sum of $ risk (distance to SL x volume) across all open
    positions on SYMBOL, as a percent of current balance. Ignores
    positions with no SL set (can't quantify their risk)."""

    positions = get_open_positions()
    if not positions:
        return 0.0

    info = mt5.symbol_info(SYMBOL)
    account = mt5.account_info()

    if info is None or account is None or account.balance <= 0:
        return 0.0

    tick_size = float(info.trade_tick_size)
    tick_value = float(info.trade_tick_value)

    if tick_size <= 0 or tick_value <= 0:
        return 0.0

    total_risk_money = 0.0

    for p in positions:
        if not p.sl:
            continue
        distance = abs(p.price_open - p.sl)
        loss_one_lot = distance / tick_size * tick_value
        total_risk_money += loss_one_lot * p.volume

    return total_risk_money / account.balance * 100


# ============================================================
# DAILY CIRCUIT BREAKER
# ============================================================

def check_daily_circuit_breaker():
    global day_start_date, day_start_balance, trading_halted, halt_reason

    account = mt5.account_info()
    if account is None:
        return

    today = time.strftime("%Y-%m-%d")

    if day_start_date != today:
        day_start_date = today
        day_start_balance = account.balance
        was_halted = trading_halted
        trading_halted = False
        halt_reason = ""
        log.info("New trading day - baseline balance %.2f", day_start_balance)
        if was_halted:
            journal_write({"event": "circuit_breaker_reset"})
        return

    if not day_start_balance or day_start_balance <= 0:
        return

    drawdown_pct = (account.balance - day_start_balance) / day_start_balance * 100

    if drawdown_pct <= -MAX_DAILY_LOSS_PERCENT and not trading_halted:
        trading_halted = True
        halt_reason = f"Daily loss limit hit ({drawdown_pct:.2f}%)"
        log.error("DAILY LOSS LIMIT HIT: %.2f%% - halting new entries", drawdown_pct)
        journal_write({
            "event": "circuit_breaker_triggered",
            "drawdown_pct": round(drawdown_pct, 2),
        })


# ============================================================
# VALIDATE AI DECISION
# ============================================================

def validate_decision(decision, state):
    action = decision["action"]

    if trading_halted:
        return False, halt_reason

    if action == "WAIT":
        return False, "WAIT"

    if action not in ["BUY", "SELL"]:
        return False, "Invalid action"

    if len(get_open_positions()) >= MAX_OPEN_TRADES:
        return False, "Maximum open trades reached"

    regime = state.get("regime", {})
    effective_min_confidence = (
        MIN_CONFIDENCE_HIGH_VOL if regime.get("volatility") == "HIGH"
        else MIN_CONFIDENCE
    )

    if decision["confidence"] < effective_min_confidence:
        return False, (
            f"Confidence below minimum "
            f"({decision['confidence']:.1f}% < {effective_min_confidence:.1f}% "
            f"required in {regime.get('volatility', 'NORMAL')} volatility)"
        )

    entry = decision["entry"]
    sl = decision["sl"]
    tp = decision["tp"]

    if entry <= 0 or sl <= 0 or tp <= 0:
        return False, "Invalid prices"

    if action == "BUY" and not (sl < entry < tp):
        return False, "Invalid BUY SL/TP"

    if action == "SELL" and not (tp < entry < sl):
        return False, "Invalid SELL SL/TP"

    atr = float(state["atr"])
    distance = abs(entry - sl)
    min_distance = atr * MIN_SL_ATR_MULT

    if distance < min_distance:
        return False, (
            f"SL too tight ({distance:.2f} < {min_distance:.2f} "
            f"= {MIN_SL_ATR_MULT}*ATR)"
        )

    open_risk = calculate_open_risk_percent()
    if open_risk + RISK_PERCENT > MAX_TOTAL_RISK_PERCENT:
        return False, (
            f"Portfolio risk cap reached ({open_risk:.2f}% open + "
            f"{RISK_PERCENT:.2f}% new > {MAX_TOTAL_RISK_PERCENT:.2f}%)"
        )

    return True, "OK"


# ============================================================
# PENDING SETUP
# ============================================================

def create_pending_setup(decision, state):
    global pending_setup

    pending_setup = {
        "action": decision["action"],
        "confidence": decision["confidence"],
        "entry": decision["entry"],
        "sl": decision["sl"],
        "tp": decision["tp"],
        "reason": decision["reason"],
        "atr": float(state["atr"]),
        "created": time.time(),
    }


def check_entry(setup):
    tick = mt5.symbol_info_tick(SYMBOL)
    info = mt5.symbol_info(SYMBOL)

    if tick is None or info is None:
        return False, 0, 0

    current = float(tick.ask) if setup["action"] == "BUY" else float(tick.bid)
    entry = float(setup["entry"])

    tolerance = max(setup["atr"] * ENTRY_TOLERANCE_ATR, float(info.point) * 10)
    distance = abs(current - entry)

    return distance <= tolerance, current, tolerance


# ============================================================
# RISK-SIZED VOLUME
# ============================================================

def calculate_volume(entry, sl, atr):
    """Returns (volume, actual_risk_percent). Raises RiskRejected
    if the setup fails a risk sanity check - this is a deliberate
    refusal to trade, not an error to swallow."""

    account = mt5.account_info()
    info = mt5.symbol_info(SYMBOL)

    if account is None or info is None:
        raise RuntimeError("MT5 account/symbol unavailable")

    distance = abs(entry - sl)

    min_distance = atr * MIN_SL_ATR_MULT
    if distance < min_distance:
        raise RiskRejected(
            f"SL too tight ({distance:.2f} < {min_distance:.2f})"
        )

    tick_size = float(info.trade_tick_size)
    tick_value = float(info.trade_tick_value)

    if tick_size <= 0 or tick_value <= 0:
        raise RuntimeError("Invalid tick information")

    loss_one_lot = distance / tick_size * tick_value
    if loss_one_lot <= 0:
        raise RuntimeError("Invalid calculated loss")

    risk_money = account.balance * RISK_PERCENT / 100
    volume = risk_money / loss_one_lot

    volume = max(float(info.volume_min), min(volume, float(info.volume_max)))

    step = float(info.volume_step)
    if step > 0:
        volume = math.floor(volume / step) * step

    volume = round(volume, 2)

    if volume <= 0:
        raise RiskRejected("Calculated volume is zero")

    actual_risk_money = volume * loss_one_lot
    actual_risk_percent = actual_risk_money / account.balance * 100

    if actual_risk_percent > RISK_PERCENT * MAX_RISK_OVERRUN_MULT:
        raise RiskRejected(
            f"Broker min lot forces {actual_risk_percent:.2f}% risk "
            f"(target {RISK_PERCENT:.2f}%)"
        )

    return volume, actual_risk_percent


def get_filling_mode():
    info = mt5.symbol_info(SYMBOL)
    if info is None:
        return mt5.ORDER_FILLING_IOC

    if info.filling_mode & mt5.ORDER_FILLING_FOK:
        return mt5.ORDER_FILLING_FOK

    if info.filling_mode & mt5.ORDER_FILLING_IOC:
        return mt5.ORDER_FILLING_IOC

    return mt5.ORDER_FILLING_IOC


def build_order(decision, volume):
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        raise RuntimeError("No market tick")

    if decision["action"] == "BUY":
        order_type = mt5.ORDER_TYPE_BUY
        price = float(tick.ask)
    else:
        order_type = mt5.ORDER_TYPE_SELL
        price = float(tick.bid)

    return {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": volume,
        "type": order_type,
        "price": price,
        "sl": decision["sl"],
        "tp": decision["tp"],
        "deviation": DEVIATION,
        "magic": MAGIC,
        "comment": "AI_TRADER_V3",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": get_filling_mode(),
    }


def execute_order(request):
    global last_order_status

    check = mt5.order_check(request)

    if check is None:
        last_order_status = "ORDER CHECK FAILED"
        log.error("ORDER CHECK FAILED: %s", mt5.last_error())
        journal_write({"event": "order_check_failed", "error": str(mt5.last_error())})
        return False

    # For order_check, retcode == 0 means the request passed and
    # can be sent. Anything else (invalid stops, not enough margin,
    # market closed, etc.) must block the send - the old code only
    # checked "is check None", which ignored this entirely.
    if check.retcode != 0:
        last_order_status = f"ORDER CHECK REJECTED | retcode={check.retcode}"
        log.error(
            "ORDER CHECK REJECTED: retcode=%s comment=%s",
            check.retcode, getattr(check, "comment", "")
        )
        journal_write({
            "event": "order_check_rejected",
            "retcode": check.retcode,
            "comment": getattr(check, "comment", ""),
        })
        return False

    if DRY_RUN:
        last_order_status = "DRY RUN - ORDER CHECK OK"
        log.info("DRY_RUN=true - order not sent")
        journal_write({"event": "dry_run_order_ok", "request": request})
        return True

    result = mt5.order_send(request)

    if result is None:
        last_order_status = "ORDER SEND FAILED"
        log.error("ORDER SEND FAILED: %s", mt5.last_error())
        journal_write({"event": "order_send_failed", "error": str(mt5.last_error())})
        return False

    last_order_status = f"ORDER SENT | retcode={result.retcode}"
    log.info("ORDER RESULT: %s", result)

    journal_write({
        "event": "order_sent",
        "retcode": result.retcode,
        "volume": request["volume"],
        "type": request["type"],
        "price": request["price"],
        "sl": request["sl"],
        "tp": request["tp"],
    })

    return True


# ============================================================
# MONITOR ENTRY
# ============================================================

def monitor_pending_entry():
    global pending_setup, last_order_status, last_error

    if pending_setup is None:
        return

    age = time.time() - pending_setup["created"]
    if age > SETUP_MAX_SECONDS:
        pending_setup = None
        last_order_status = "AI SETUP EXPIRED"
        return

    if len(get_open_positions()) >= MAX_OPEN_TRADES:
        pending_setup = None
        last_order_status = "MAX TRADES REACHED"
        return

    reached, price, tolerance = check_entry(pending_setup)
    if not reached:
        last_order_status = "WAITING FOR ENTRY"
        return

    action = pending_setup["action"]
    sl = pending_setup["sl"]
    tp = pending_setup["tp"]

    if action == "BUY" and not (sl < price < tp):
        pending_setup = None
        last_order_status = "CANCELLED - INVALID BUY PRICE"
        return

    if action == "SELL" and not (tp < price < sl):
        pending_setup = None
        last_order_status = "CANCELLED - INVALID SELL PRICE"
        return

    # Re-check the portfolio risk cap right before execution - the
    # setup was created up to 15 minutes ago, other trades may have
    # opened since.
    open_risk = calculate_open_risk_percent()
    if open_risk + RISK_PERCENT > MAX_TOTAL_RISK_PERCENT:
        pending_setup = None
        last_order_status = "CANCELLED - PORTFOLIO RISK CAP"
        journal_write({"event": "setup_cancelled", "reason": "portfolio_risk_cap"})
        return

    try:
        volume, actual_risk_pct = calculate_volume(
            price, sl, pending_setup["atr"]
        )
    except RiskRejected as e:
        last_error = str(e)
        pending_setup = None
        last_order_status = f"SETUP REJECTED - {e}"
        journal_write({"event": "setup_rejected", "reason": str(e)})
        return
    except Exception as e:
        last_error = str(e)
        pending_setup = None
        last_order_status = "VOLUME CALCULATION ERROR"
        journal_write({"event": "volume_error", "reason": str(e)})
        return

    decision = {
        "action": pending_setup["action"],
        "confidence": pending_setup["confidence"],
        "entry": pending_setup["entry"],
        "sl": pending_setup["sl"],
        "tp": pending_setup["tp"],
        "reason": pending_setup["reason"],
    }

    try:
        request = build_order(decision, volume)
        pending_setup = None
        execute_order(request)
    except Exception as e:
        last_error = str(e)
        pending_setup = None
        last_order_status = "EXECUTION ERROR"
        journal_write({"event": "execution_error", "reason": str(e)})


# ============================================================
# CLOSED-TRADE JOURNAL
# ============================================================

def journal_closed_deals():
    """Logs realized P/L for every closed deal with our MAGIC
    number, so you can compute win rate / R / edge later from
    the journal file. Safe to call every loop iteration."""

    now = datetime.datetime.now()
    from_dt = now - datetime.timedelta(days=2)

    deals = mt5.history_deals_get(from_dt, now)
    if deals is None:
        return

    for d in deals:
        if d.magic != MAGIC:
            continue
        if d.entry != mt5.DEAL_ENTRY_OUT:
            continue
        if d.ticket in seen_deal_tickets:
            continue

        seen_deal_tickets.add(d.ticket)

        journal_write({
            "event": "trade_closed",
            "ticket": d.ticket,
            "symbol": d.symbol,
            "volume": d.volume,
            "price": d.price,
            "profit": d.profit,
            "swap": d.swap,
            "commission": d.commission,
        })


# ============================================================
# ANALYZE NEW M15 BAR
# ============================================================

def analyze_market():
    global pending_setup, last_ai_decision, last_analysis_time
    global last_team_results, last_order_status, last_error
    global last_market_state

    last_error = ""
    pending_setup = None
    last_order_status = "AI ANALYZING..."

    df = get_market()

    try:
        htf = get_htf_context()
    except Exception as e:
        htf = None
        log.info("HTF context fetch failed: %s", e)

    state = build_market_state(df, htf)
    last_market_state = state

    chart_b64 = None
    if USE_CHART_VISION:
        try:
            chart_b64 = render_chart_image(df)
        except Exception as e:
            last_error = str(e)
            log.info("Chart render failed: %s", e)

    analyses = run_team(state, chart_b64)
    last_team_results = analyses

    decision = run_judge(state, analyses, chart_b64)
    last_ai_decision = decision
    last_analysis_time = time.strftime("%Y-%m-%d %H:%M:%S")

    journal_write({
        "event": "analysis",
        "decision": decision,
        "team": [
            {"action": a["action"], "confidence": a["confidence"]}
            for a in analyses
        ],
    })

    if decision["action"] == "WAIT":
        pending_setup = None
        last_order_status = "AI SAYS WAIT"
        return

    valid, reason = validate_decision(decision, state)
    if not valid:
        pending_setup = None
        last_order_status = f"SETUP BLOCKED: {reason}"
        journal_write({"event": "setup_blocked", "reason": reason})
        return

    create_pending_setup(decision, state)
    last_order_status = "WAITING FOR AI ENTRY"


def is_new_bar():
    global last_analysis_bar

    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 1)
    if rates is None or len(rates) == 0:
        return False

    current_bar = rates[0]["time"]

    if last_analysis_bar is None:
        last_analysis_bar = current_bar
        return True

    if current_bar != last_analysis_bar:
        last_analysis_bar = current_bar
        return True

    return False


# ============================================================
# MAIN
# ============================================================

def main():
    global last_error, last_order_status

    connect_mt5()
    last_order_status = "SYSTEM READY"

    try:
        analyze_market()
    except Exception as e:
        last_error = str(e)
        last_order_status = "INITIAL ANALYSIS ERROR"

    while True:
        try:
            check_daily_circuit_breaker()

            if is_new_bar():
                analyze_market()

            monitor_pending_entry()
            journal_closed_deals()
            print_screen()

        except Exception as e:
            last_error = str(e)
            last_order_status = "MAIN LOOP ERROR"
            log.exception("MAIN LOOP ERROR")

        time.sleep(LOOP_SECONDS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print("AI TRADER STOPPED")
    except Exception as e:
        print()
        print("FATAL ERROR:")
        print(e)
    finally:
        mt5.shutdown()
