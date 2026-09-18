"""
NSE Daily Screener Agent
Scans Nifty 500 -> technical + volume + fundamentals + historical-analog score -> Top 5 shortlist.
Research tool only. Not investment advice.

Run:  python nse_agent.py
Env (optional, for email): EMAIL_USER, EMAIL_APP_PASSWORD, EMAIL_TO
"""
import os, io, smtplib, datetime as dt
from email.mime.text import MIMEText
import numpy as np
import pandas as pd
import requests
import yfinance as yf

UNIVERSE_URL = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
MIN_TRADED_VALUE = 10e7      # Rs 10 Cr average daily traded value
TOP_N = 5
FWD_DAYS = 20                # ~4 weeks forward window for historical analogs
HEADERS = {"User-Agent": "Mozilla/5.0"}


def load_universe():
    try:
        r = requests.get(UNIVERSE_URL, headers=HEADERS, timeout=30)
        r.raise_for_status()
        return [s + ".NS" for s in pd.read_csv(io.StringIO(r.text))["Symbol"]]
    except Exception as e:
        print("Universe download failed, using universe.txt:", e)
        with open("universe.txt") as f:  # one NSE symbol per line, e.g. RELIANCE
            return [l.strip() + ".NS" for l in f if l.strip()]


def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def add_indicators(df):
    c, v = df["Close"], df["Volume"]
    df["dma20"], df["dma50"], df["dma200"] = (c.rolling(w).mean() for w in (20, 50, 200))
    df["rsi"] = rsi(c)
    ema12, ema26 = c.ewm(span=12).mean(), c.ewm(span=26).mean()
    macd = ema12 - ema26
    df["macd_hist"] = macd - macd.ewm(span=9).mean()
    df["vol_ratio"] = v / v.rolling(20).mean()
    df["hi20"] = c.rolling(20).max()
    df["hi252"] = c.rolling(252).max()
    df["traded_val"] = (c * v).rolling(20).mean()
    return df


def setup_mask(df):
    """Bullish setup definition, reused for today's scan AND historical analogs."""
    return (
        (df["Close"] > df["dma50"]) & (df["dma50"] > df["dma200"])
        & df["rsi"].between(55, 72) & (df["macd_hist"] > 0)
        & (df["Close"] >= 0.97 * df["hi20"]) & (df["vol_ratio"] >= 1.2)
    )


def historical_analog(df):
    """How did THIS stock behave after similar setups in past ~5y?"""
    fwd = df["Close"].shift(-FWD_DAYS) / df["Close"] - 1
    hits = fwd[setup_mask(df)].dropna()
    if len(hits) < 3:
        return np.nan, np.nan, len(hits)
    return (hits > 0.03).mean(), hits.median(), len(hits)


def fundamentals(sym):
    try:
        i = yf.Ticker(sym).info
    except Exception:
        return {}
    return {
        "roe": i.get("returnOnEquity"), "de": i.get("debtToEquity"),
        "rev_g": i.get("revenueGrowth"), "earn_g": i.get("earningsGrowth"),
        "pe": i.get("trailingPE"), "sector": i.get("sector"),
    }


def tech_score(r):
    s = 0
    s += 8 if r.Close > r.dma50 > r.dma200 else 0
    s += 5 if r.dma20 > r.dma50 else 0
    s += 6 if 55 <= r.rsi <= 70 else (2 if r.rsi < 55 else 0)
    s += 6 if r.macd_hist > 0 else 0
    s += 5 if r.Close >= 0.97 * r.hi20 else 0
    s += 5 if r.Close >= 0.85 * r.hi252 else 0
    return s  # max 35


def vol_score(r):
    return float(np.clip((r.vol_ratio - 1) / 1.0, 0, 1) * 20)  # max 20


def fund_score(f):
    if not f:
        return 5
    s = 0
    s += 7 if (f.get("roe") or 0) > 0.15 else (3 if (f.get("roe") or 0) > 0.10 else 0)
    s += 5 if f.get("de") is not None and f["de"] < 100 else 0   # yfinance gives D/E in %
    s += 4 if (f.get("rev_g") or 0) > 0.10 else 0
    s += 4 if (f.get("earn_g") or 0) > 0.10 else 0
    return s  # max 20


def analog_score(win, med):
    if np.isnan(win):
        return 5
    return float(np.clip(win, 0, 1) * 15 + np.clip(med / 0.08, 0, 1) * 10)  # max 25


def main():
    syms = load_universe()
    print(f"Scanning {len(syms)} stocks...")
    data = yf.download(syms, period="6y", interval="1d", group_by="ticker",
                       auto_adjust=True, threads=True, progress=False)
    rows = []
    for sym in syms:
        try:
            df = data[sym].dropna().copy()
            if len(df) < 260:
                continue
            df = add_indicators(df)
            r = df.iloc[-1]
            if r.traded_val < MIN_TRADED_VALUE or r.Close < 20:
                continue
            if not (r.Close > r.dma50 > r.dma200):      # hard trend filter
                continue
            win, med, n = historical_analog(df)
            rows.append(dict(sym=sym, close=r.Close, rsi=r.rsi, vol_ratio=r.vol_ratio,
                             win=win, med=med, n=n,
                             tech=tech_score(r), vol=vol_score(r),
                             stop=min(df["Low"].tail(10).min(), r.dma20) * 0.99,
                             atr=(df["High"] - df["Low"]).tail(14).mean()))
        except Exception:
            continue

    res = pd.DataFrame(rows)
    if res.empty:
        print("No candidates today.")
        return
    # Fundamentals only for the top 30 technical candidates (saves time)
    res["pre"] = res["tech"] + res["vol"]
    res = res.sort_values("pre", ascending=False).head(30).copy()
    funds = {s: fundamentals(s) for s in res["sym"]}
    res["fund"] = [fund_score(funds[s]) for s in res["sym"]]
    res["analog"] = [analog_score(w, m) for w, m in zip(res["win"], res["med"].fillna(0))]
    res["score"] = res["tech"] + res["vol"] + res["fund"] + res["analog"]   # max 100
    res["sector"] = [funds[s].get("sector") for s in res["sym"]]

    # Diversify: max 2 per sector
    top, cnt = [], {}
    for _, r in res.sort_values("score", ascending=False).iterrows():
        if cnt.get(r["sector"], 0) < 2:
            top.append(r); cnt[r["sector"]] = cnt.get(r["sector"], 0) + 1
        if len(top) == TOP_N:
            break
    out = pd.DataFrame(top)
    out["entry_zone"] = out["close"].round(1).astype(str) + " - " + (out["close"] * 1.01).round(1).astype(str)
    out["stop_loss"] = out["stop"].round(1)
    out["target_1"] = (out["close"] + 2 * (out["close"] - out["stop"])).round(1)   # 1:2 risk-reward
    out["hist_win%"] = (out["win"] * 100).round(0)
    out["hist_median%"] = (out["med"] * 100).round(1)
    cols = ["sym", "score", "entry_zone", "stop_loss", "target_1", "rsi", "vol_ratio",
            "hist_win%", "hist_median%", "n", "sector"]
    out = out[cols].round(1)

    today = dt.date.today().isoformat()
    out.to_csv(f"top5_{today}.csv", index=False)
    text = (f"NSE Top {TOP_N} shortlist - {today}\n\n{out.to_string(index=False)}\n\n"
            "Screening output only, not advice. Use stop-loss, size positions small, "
            "do not chase gap-ups. Verify before acting.")
    print(text)
    send_email(f"NSE Top 5 - {today}", text)


def send_email(subject, body):
    u, p, to = (os.getenv(k) for k in ("EMAIL_USER", "EMAIL_APP_PASSWORD", "EMAIL_TO"))
    if not (u and p and to):
        return
    m = MIMEText(body); m["Subject"], m["From"], m["To"] = subject, u, to
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(u, p); s.send_message(m)
    print("Email sent.")


if __name__ == "__main__":
    main()
