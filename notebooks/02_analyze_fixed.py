# 02 — Analyze, Visualize, Draft (fixed)
from pathlib import Path
import json, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import unicodedata

PROJ = Path.cwd()
DATA = PROJ / "data"
RAW  = DATA / "raw"
PROC = DATA / "processed"
OUT  = DATA / "outputs"
for p in (DATA, RAW, PROC, OUT): p.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = RAW / "config.json"
CLEAN_PATH  = PROC / "posts_clean.parquet"

cfg = json.load(open(CONFIG_PATH, "r", encoding="utf-8"))
SYMBOLS = cfg["symbols"]
PHRASES = cfg["phrases"]
SINCE_ISO = cfg.get("since_iso")
TARGET = int(cfg["target_posts_per_pub"])

df = pd.read_parquet(CLEAN_PATH).copy()

def _canon(t: str) -> str:
    if not isinstance(t, str): return ""
    t = unicodedata.normalize("NFKC", t)
    for ch in ["\uFE0F", "\uFE0E", "\u200D"]:
        t = t.replace(ch, "")
    t = t.replace("–", "—").replace("--", "—")
    return " ".join(t.split())

df["text"] = df["text"].astype(str).map(_canon)
df["published"] = pd.to_datetime(df["published"], utc=True)

ALIASES = {
    "green_check": ["✅", "✔", "✔️"],
    "ellipsis": ["…", "..."],
}
DASH_RE = re.compile(r"(?<=\w)\s*[—–-]{1,2}\s*(?=\w)")

def rate_per_1k_literal(text: str, token: str) -> float:
    if not text: return 0.0
    n = text.count(token)
    return n / max(len(text) / 1000.0, 1e-9)

def rate_per_1k_any(text: str, tokens: list) -> float:
    if not text: return 0.0
    n = sum(text.count(tok) for tok in tokens)
    return n / max(len(text) / 1000.0, 1e-9)

def rate_per_1k_dashlike(text: str) -> float:
    if not text: return 0.0
    n = len(DASH_RE.findall(text))
    return n / max(len(text) / 1000.0, 1e-9)

def rate_per_1k_regex(text: str, pattern: str) -> float:
    if not text: return 0.0
    n = len(re.findall(pattern, text, flags=re.I))
    return n / max(len(text) / 1000.0, 1e-9)

# Build metrics
metrics = {}
metrics["dashlike_per1k"] = df["text"].apply(rate_per_1k_dashlike)
for label, tok in SYMBOLS.items():
    if label == "emdash":
        metrics[f"{label}_per1k"] = df["text"].apply(lambda t: rate_per_1k_literal(t, tok))
    else:
        tokens = ALIASES.get(label, [tok])
        metrics[f"{label}_per1k"] = df["text"].apply(lambda t: rate_per_1k_any(t, tokens))
for i, pat in enumerate(PHRASES):
    metrics[f"phrase_{i}_per1k"] = df["text"].apply(lambda t, p=pat: rate_per_1k_regex(t, p))

df_metrics = df.assign(**metrics)

# ---- Month bucketing + last-two-years (timezone-safe) ----
# Make published timestamps UTC-aware for comparisons
pub_aware = pd.to_datetime(df_metrics["published"], utc=True)

# Build a UTC-aware 'since' from SINCE_ISO (fallback: 24 months before latest pub)
if SINCE_ISO:
    since = pd.Timestamp(SINCE_ISO)
else:
    latest = pub_aware.max()
    since = (latest - pd.Timedelta(days=730))
if since.tz is None:
    since = since.tz_localize("UTC")
else:
    since = since.tz_convert("UTC")

# Filter last two years using tz-aware comparison
df_metrics_2y = df_metrics[pub_aware >= since].copy()

# Month labels should be tz-naive strings; drop tz after filtering
df_metrics["month"] = pub_aware.dt.tz_localize(None).dt.to_period("M").astype(str)
pub_aware_2y = pd.to_datetime(df_metrics_2y["published"], utc=True)
df_metrics_2y["month"] = pub_aware_2y.dt.tz_localize(None).dt.to_period("M").astype(str)

# ---- Per-pub means table ----
value_cols = [c for c in df_metrics.columns if c.endswith("_per1k")]
agg_pub = df_metrics.groupby("pub")[value_cols].mean().sort_values(
    "rocket_per1k" if "rocket_per1k" in value_cols else value_cols[0],
    ascending=False
)
agg_pub_rounded = agg_pub.round(3)
print("Metric columns:", value_cols)
print(agg_pub_rounded.head())

# ---- Save bar chart ----
plt.figure()
col = "rocket_per1k" if "rocket_per1k" in agg_pub.columns else value_cols[0]
(agg_pub[col].sort_values(ascending=True)).plot(kind="barh", title=f"{col} (mean by publication)")
plt.xlabel("rate per 1k chars")
plt.tight_layout()
plt.savefig(OUT / "rockets_by_pub.png", dpi=200)
plt.close()

# ---- Save heatmap ----
plt.figure()
z = (agg_pub - agg_pub.mean()) / agg_pub.std(ddof=0).replace(0, 1.0)
plt.imshow(z.values, aspect="auto")
plt.yticks(range(len(z.index)), z.index)
plt.xticks(range(len(z.columns)), z.columns, rotation=45, ha="right")
plt.title("Symbol & phrase z-scores by publication")
plt.colorbar()
plt.tight_layout()
plt.savefig(OUT / "workslop_heatmap.png", dpi=200)
plt.close()

# =========================
# Monthly growth charts (past two years)
# Weighted by text length for accurate global rates
# =========================

tokens_to_plot = [
    "emdash_per1k", "dashlike_per1k",
    "rocket_per1k", "green_check_per1k", "sparkles_per1k",
    "fire_per1k", "robot_per1k", "chart_up_per1k", "ellipsis_per1k"
]
tokens_to_plot = [c for c in tokens_to_plot if c in value_cols]

# Helper: build weighted monthly rate (sum counts / sum kchars)
def monthly_weighted_rate(df_in, rate_col):
    if df_in.empty:
        return pd.Series(dtype=float)
    kchars = df_in["n_chars"].astype(float) / 1000.0
    counts = df_in[rate_col].astype(float) * kchars
    grp = pd.DataFrame({"counts": counts, "kchars": kchars, "month": df_in["month"]})
    bym = grp.groupby("month", sort=True).sum()
    bym = bym[bym["kchars"] > 0]
    return (bym["counts"] / bym["kchars"]).sort_index()

# ---- Overall (all pubs) monthly trends ----
overall_trends = {}
for c in tokens_to_plot:
    overall_trends[c] = monthly_weighted_rate(df_metrics_2y, c)

# Align on common month index
if overall_trends:
    all_idx = sorted(set().union(*[s.index for s in overall_trends.values()]))
    trends_df = pd.DataFrame({k: overall_trends[k].reindex(all_idx) for k in overall_trends}).fillna(0.0)
    trends_df.index.name = "month"

    plt.figure(figsize=(11, 6))
    for c in trends_df.columns:
        plt.plot(trends_df.index, trends_df[c], label=c.replace("_per1k", ""))
    plt.title("Workslop characters & emojis — monthly rate per 1k chars (last 2 years, overall)")
    plt.xlabel("Month")
    plt.ylabel("Rate per 1k chars")
    plt.xticks(rotation=45, ha="right")
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT / "workslop_trends_overall.png", dpi=200)
    plt.close()

# ---- Per-publication monthly trend for 🚀 (top 5 pubs by volume) ----
pub_volume = df_metrics_2y.groupby("pub")["n_chars"].sum().sort_values(ascending=False)
top5 = list(pub_volume.head(5).index)

plt.figure(figsize=(11, 6))
for pub in top5:
    sub = df_metrics_2y[df_metrics_2y["pub"] == pub]
    s = monthly_weighted_rate(sub, "rocket_per1k") if "rocket_per1k" in value_cols else pd.Series(dtype=float)
    if not s.empty:
        plt.plot(s.index, s.values, label=pub)

plt.title("🚀 rocket — monthly rate per 1k chars (last 2 years, top 5 pubs by volume)")
plt.xlabel("Month")
plt.ylabel("Rate per 1k chars")
plt.xticks(rotation=45, ha="right")
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig(OUT / "workslop_trends_top5pubs_rocket.png", dpi=200)
plt.close()

print("Wrote trend figures to", OUT)
