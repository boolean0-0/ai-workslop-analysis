# 00 — Config & Targets (fixed)
from pathlib import Path
import json, datetime as dt

PROJ = Path.cwd()
DATA = PROJ / "data"
RAW  = DATA / "raw"
PROC = DATA / "processed"
OUT  = DATA / "outputs"
for p in (DATA, RAW, PROC, OUT): p.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = RAW / "config.json"

SUBSTACKS = {
    "Read Max": "https://maxread.substack.com/feed",
    "Noahpinion": "https://noahpinion.substack.com/feed",
    "The Intrinsic Perspective": "https://erikhoel.substack.com/feed",
    "Cliodynamica by Peter Turchin": "https://peterturchin.substack.com/feed",
    "The Culturist": "https://culturist.substack.com/feed",
    "Story Club by George Saunders": "https://georgesaunders.substack.com/feed",
    "Poetic Outlaws": "https://poeticoutlaws.substack.com/feed",
    "Hardware FYI": "https://hardwarefyi.substack.com/feed",
    "Letters from an American by Heather Cox Richardson": "https://heathercoxrichardson.substack.com/feed",
    "Anton Howes": "https://antonhowes.substack.com/feed",
}


START_DATE = "2023-01-01"
END_DATE   = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
YEARS_BACK = 2
SINCE = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=365 * YEARS_BACK)).isoformat()

TARGET_POSTS_PER_PUB = 50

SYMBOLS = {
    "emdash": "—",
    "rocket": "🚀",
    "green_check": "✅",
    "sparkles": "✨",
    "fire": "🔥",
    "robot": "🤖",
    "chart_up": "📈",
    "ellipsis": "…",
}

PHRASES = [
    r"\bwe'?re\s+excited\s+to\s+announce\b",
]

REQUEST_SLEEP_S = 0.2
REQUEST_TIMEOUT_S = 10

cfg = {
    "subscribes": SUBSTACKS,
    "start_date": START_DATE,
    "end_date": END_DATE,
    "since_iso": SINCE,
    "years_back": YEARS_BACK,
    "target_posts_per_pub": TARGET_POSTS_PER_PUB,
    "symbols": SYMBOLS,
    "phrases": PHRASES,
    "request_sleep_s": REQUEST_SLEEP_S,
    "request_timeout_s": REQUEST_TIMEOUT_S,
}

with open(CONFIG_PATH, "w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=2)

print(f"Wrote {CONFIG_PATH}")