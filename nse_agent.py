"""
NSE Daily Screener Agent
Scans Nifty 500 -> technical + volume + fundamentals + historical-analog score -> Top 5 shortlist.
Research tool only. Not investment advice.

Run:  python nse_agent.py
Env (optional, for email): EMAIL_USER, EMAIL_APP_PASSWORD, EMAIL_TO
"""
import os, io, smtplib, datetime as dt
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import html as _html
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
    try:
        save_to_sheet(out, today)
    except Exception as e:
        print("Sheet save failed (email will still be sent):", e)
    send_email(f"NSE Top 5 - {today}", text, build_html(out, today))


def _fmt(v, suffix="", dash="-"):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return dash
        return f"{v:g}{suffix}"
    except Exception:
        return str(v)


def build_html(out, today):
    """Banner + table style HTML email (inline CSS so Gmail/Outlook render it)."""
    d = dt.date.fromisoformat(today).strftime("%A, %d %b %Y")
    rows = ""
    for i, r in enumerate(out.itertuples(index=False), 1):
        name = _html.escape(str(r.sym).replace(".NS", ""))
        sector = _html.escape(str(r.sector) if isinstance(r.sector, str) and r.sector else "-")
        bg = "#ffffff" if i % 2 else "#f5f8fc"
        sc = float(r.score)
        med_color = ('#15803d' if (isinstance(r[8], (int, float)) and not np.isnan(r[8]) and r[8] > 0)
                     else ('#b91c1c' if (isinstance(r[8], (int, float)) and not np.isnan(r[8])) else '#6b7280'))
        badge = "#16a34a" if sc >= 70 else ("#d97706" if sc >= 55 else "#6b7280")
        td = f"padding:12px 8px;border-bottom:1px solid #e5e7eb;font-size:13px;text-align:center;background:{bg};"
        rows += (
            f"<tr>"
            f"<td style='{td}font-weight:bold;color:#0b3d91;'>{i}</td>"
            f"<td style='{td}text-align:left;'>"
            f"<a href='https://www.tradingview.com/chart/?symbol=NSE:{name}' "
            f"style='color:#0b3d91;font-weight:bold;font-size:15px;text-decoration:none;'>{name}</a>"
            f"<br><span style='color:#6b7280;font-size:11px;'>{sector}</span></td>"
            f"<td style='{td}'><span style='background:{badge};color:#fff;padding:3px 9px;"
            f"border-radius:10px;font-weight:bold;'>{_fmt(sc)}</span></td>"
            f"<td style='{td}'>{_html.escape(str(r.entry_zone))}</td>"
            f"<td style='{td}color:#b91c1c;font-weight:bold;'>{_fmt(r.stop_loss)}</td>"
            f"<td style='{td}color:#15803d;font-weight:bold;'>{_fmt(r.target_1)}</td>"
            f"<td style='{td}'>{_fmt(r.rsi)}</td>"
            f"<td style='{td}'>{_fmt(r.vol_ratio, 'x')}</td>"
            f"<td style='{td}'>{_fmt(r[7], '%')}</td>"
            f"<td style='{td}color:{med_color};font-weight:bold;'>{_fmt(r[8], '%')}</td>"
            f"<td style='{td}color:#6b7280;'>{_fmt(r[9])}</td>"
            f"</tr>"
        )
    th = ("padding:10px 8px;font-size:11px;color:#ffffff;background:#0b3d91;"
          "text-align:center;text-transform:uppercase;letter-spacing:.4px;")
    heads = ["#", "Stock", "Score", "Entry Zone", "Stop Loss", "Target", "RSI", "Vol", "Hist Win", "Hist Median", "n"]
    head_html = "".join(
        f"<th style='{th}{'text-align:left;' if h == 'Stock' else ''}'>{h}</th>" for h in heads)
    return f"""<!DOCTYPE html>
<html><body style="margin:0;padding:0;background:#eef2f7;font-family:Arial,Helvetica,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#eef2f7;padding:16px 0;"><tr><td align="center">
<table width="760" cellpadding="0" cellspacing="0" style="max-width:760px;width:100%;background:#ffffff;border-radius:10px;overflow:hidden;">
  <tr><td style="background:#0b3d91;padding:26px 22px;text-align:center;">
      <div style="color:#fbbf24;font-size:12px;letter-spacing:2px;font-weight:bold;">DAILY SCREENER REPORT</div>
      <div style="color:#ffffff;font-size:26px;font-weight:bold;margin:6px 0;">NSE Top 5 Stocks Shortlist</div>
      <div style="color:#cbd5e1;font-size:13px;">{d} &nbsp;|&nbsp; Nifty 500 scan</div>
  </td></tr>
  <tr><td style="background:#fbbf24;padding:8px;text-align:center;font-size:12px;color:#1f2937;font-weight:bold;">
      Trend + Volume + Fundamentals + Historical Pattern Score (out of 100)
  </td></tr>
  <tr><td style="padding:16px 10px;">
    <div style="overflow-x:auto;">
    <table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;min-width:640px;">
      <tr>{head_html}</tr>
      {rows}
    </table></div>
  </td></tr>
  <tr><td style="padding:4px 22px 18px;font-size:12px;color:#374151;line-height:1.6;">
      <b>How to use:</b> Enter only inside the entry zone (do not chase gap-ups), always place the stop-loss,
      keep each stock at 15-20% of portfolio at most. Tap a stock name to open its chart.
  </td></tr>
  <tr><td style="background:#f3f4f6;padding:14px 22px;font-size:11px;color:#6b7280;text-align:center;line-height:1.5;">
      Automated screening output for research only. Not investment advice. Past patterns do not guarantee future returns.
      Verify before acting.
  </td></tr>
</table></td></tr></table>
</body></html>"""


def save_to_sheet(out, today):
    """Append today's Top 5 rows to Google Sheet through an Apps Script web app."""
    url, token = os.getenv("SHEET_WEBHOOK_URL"), os.getenv("SHEET_TOKEN")
    if not (url and token):
        print("Sheet webhook not configured, skipping.")
        return

    def clean(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        return v.item() if hasattr(v, "item") else v

    rows = []
    for r in out.itertuples(index=False):
        # Sheet order after Sr. No. (A) :
        # B Date | C Stock | D Industries | E Score | F Entry Zone | G Stop Loss | H Target
        # I RSI | J Volume | K Hist Win | L Hist Median % | M n
        rows.append([today, str(r.sym).replace(".NS", ""), clean(r.sector), clean(r.score),
                     str(r.entry_zone), clean(r.stop_loss), clean(r.target_1),
                     clean(r.rsi), clean(r.vol_ratio),
                     clean(r[7]), clean(r[8]), clean(r[9])])   # r[7]=hist_win%, r[8]=hist_median%, r[9]=n
    resp = requests.post(url, json={"token": token, "rows": rows}, timeout=60)
    print("Sheet response:", resp.text[:200])


def send_email(subject, body, html_body=None):
    u, p, to = (os.getenv(k) for k in ("EMAIL_USER", "EMAIL_APP_PASSWORD", "EMAIL_TO"))
    if not (u and p and to):
        return
    m = MIMEMultipart("alternative")
    m["Subject"], m["From"], m["To"] = subject, u, to
    m.attach(MIMEText(body, "plain"))
    if html_body:
        m.attach(MIMEText(html_body, "html"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(u, p); s.send_message(m)
    print("Email sent.")


if __name__ == "__main__":
    main()
