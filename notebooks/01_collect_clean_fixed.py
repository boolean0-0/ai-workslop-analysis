# 01 — Collect (RSS) & Clean (fixed)
from pathlib import Path
import json, time, hashlib, datetime as dt, math, random, unicodedata, re
from urllib.parse import urlparse
import pandas as pd
import feedparser, requests
from bs4 import BeautifulSoup
from html import unescape

# ---------- Paths & Config ----------
PROJ = Path.cwd()
DATA = PROJ / "data"
RAW  = DATA / "raw"
PROC = DATA / "processed"
OUT  = DATA / "outputs"
for p in (DATA, RAW, PROC, OUT): p.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = RAW / "config.json"
CACHE_DIR = RAW / "cache"; CACHE_DIR.mkdir(parents=True, exist_ok=True)

META_PATH = RAW / "posts_meta.parquet"
CLEAN_PATH = PROC / "posts_clean.parquet"

cfg = json.load(open(CONFIG_PATH, "r", encoding="utf-8"))
SUBS = cfg["subscribes"]
SYMBOLS = cfg["symbols"]
START_DATE = pd.Timestamp(cfg.get("start_date", "2023-01-01"))
END_DATE = pd.Timestamp(cfg.get("end_date"))
TARGET = int(cfg["target_posts_per_pub"])
SLEEP_S = float(cfg["request_sleep_s"])
TIMEOUT_S = float(cfg["request_timeout_s"])
SINCE_ISO = cfg.get("since_iso")
YEARS_BACK = int(cfg.get("years_back", 2))
random.seed(42)

# ---------- Helpers ----------
def safe_dt_from_entry(e):
    tup = getattr(e, "published_parsed", None) or getattr(e, "updated_parsed", None)
    if tup:
        try:
            return dt.datetime(*tup[:6], tzinfo=dt.timezone.utc)
        except Exception:
            return None
    return None

def cache_path_for(url: str) -> Path:
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{h}.html"

def fetch_html(url: str) -> str:
    p = cache_path_for(url)
    if p.exists():
        try:
            return p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            pass
    try:
        r = requests.get(url, timeout=TIMEOUT_S, headers={"User-Agent": "Mozilla/5.0 (Workslop-Audit)"})
        r.raise_for_status()
        enc = r.encoding or r.apparent_encoding or "utf-8"
        html = r.content.decode(enc, errors="ignore")
        p.write_text(html, encoding="utf-8", errors="ignore")
        time.sleep(SLEEP_S)
        return html
    except Exception:
        return ""

def html_to_text(html: str) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "lxml")
    for img in soup.find_all("img"):
        rep = img.get("alt") or img.get("aria-label") or img.get("title") or ""
        img.replace_with(rep)
    for sp in soup.find_all("span"):
        rep = sp.get("aria-label") or sp.get("data-emoji-char")
        if rep:
            sp.replace_with(rep)
    # prefer common containers
    for sel in ["article", "[data-testid='post-content']", ".post", ".entry-content", ".content", ".post-content"]:
        node = soup.select_one(sel)
        if node:
            return node.get_text(" ", strip=True)
    return soup.get_text(" ", strip=True)

def rss_content_to_text(entry) -> str:
    content_html = ""
    if hasattr(entry, "content") and entry.content:
        try:
            content_html = " ".join(unescape(c.value) for c in entry.content if hasattr(c, "value"))
        except Exception:
            pass
    if (not content_html) and hasattr(entry, "summary"):
        content_html = unescape(entry.summary)
    if content_html and ("<" in content_html and ">" in content_html):
        return html_to_text(content_html)
    return (content_html or "").strip()

def normalize_text(text: str) -> str:
    """
    Canonicalize text so literal counts work:
    - Unicode NFKC
    - remove emoji VS15/VS16 and ZWJ that break literal matches
    - standardize dashes to em dash
    - collapse whitespace
    """
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    for ch in ["\uFE0F", "\uFE0E", "\u200D"]:
        t = t.replace(ch, "")
    t = t.replace("–", "—").replace("--", "—")
    return " ".join(t.split())

DASH_RE = re.compile(r"(?<=\w)\s*[—–-]{1,2}\s*(?=\w)")

def symbol_zero_check(text: str) -> bool:
    if not text:
        return True
    # Check all target symbols literally
    try:
        return all(text.count(sym) == 0 for sym in SYMBOLS.values())
    except Exception:
        return True

# ---------- Substack archive augmentation (to break past ~20 RSS cap) ----------
def _base_from_feed_url(feed_url: str) -> str:
    """Return scheme+netloc base URL for a given feed URL and strip trailing '/feed'."""
    try:
        parsed = urlparse(feed_url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        if feed_url.rstrip("/").endswith("/feed"):
            return base
        return base
    except Exception:
        return feed_url

def _parse_substack_archive_json(obj, base_url: str) -> list:
    """Parse Substack archive JSON structure into list of rows compatible with per_pub entries."""
    rows = []
    if not obj:
        return rows
    # API may return a dict with 'posts' or a raw list
    posts = obj.get("posts") if isinstance(obj, dict) else (obj if isinstance(obj, list) else [])
    for it in posts:
        try:
            url = it.get("canonical_url") or it.get("url")
            if not url:
                slug = it.get("slug")
                if slug:
                    url = f"{base_url.rstrip('/')}/p/{slug}"
            title = (it.get("title") or it.get("subject") or "").strip()
            # Common datetime keys across deployments
            dt_key = (
                it.get("post_date")
                or it.get("published_at")
                or it.get("publish_date")
                or it.get("created_at")
                or it.get("createdAt")
            )
            ts = pd.to_datetime(dt_key, utc=True, errors="coerce") if dt_key else pd.NaT
            if not url or pd.isna(ts):
                continue
            rows.append({
                "title": title,
                "url": url,
                "published": ts.to_pydatetime(),
                "rss_text": "",  # we'll hydrate from HTML in the clean step
            })
        except Exception:
            continue
    return rows

def fetch_substack_archive(feed_url: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp, target: int) -> list:
    """
    Try to fetch additional posts via Substack's archive API to go beyond the ~20-item RSS cap.
    Returns a list of dicts with keys: title, url, published, rss_text (may be empty).
    """
    base = _base_from_feed_url(feed_url)
    # Heuristic: only attempt for hosts likely backed by Substack
    if not any(h in base for h in ("substack.com", ".substack.com", ".substack", "thevccorner.com", "aidisruption.ai")):
        # Still attempt generically; fail gracefully
        pass

    results = []
    # Iterate offsets until we pass the start_dt boundary (collect as many as possible up to 2023)
    offset = 0
    page_size = 25
    max_pages = 400  # safety cap; we'll break when we cross before start_dt
    headers = {"User-Agent": "Mozilla/5.0 (Workslop-Audit)"}
    for _ in range(max_pages):
        url = f"{base.rstrip('/')}/api/v1/archive?sort=new&search=&offset={offset}&limit={page_size}"
        try:
            r = requests.get(url, timeout=TIMEOUT_S, headers=headers)
            if r.status_code != 200:
                break
            data = r.json()
            rows = _parse_substack_archive_json(data, base)
            if not rows:
                break
            # Determine the oldest timestamp on this page (rows are new->old)
            page_oldest = None
            for row in rows:
                ts_naive = pd.Timestamp(row["published"]).tz_convert(None)
                if page_oldest is None or ts_naive < page_oldest:
                    page_oldest = ts_naive
                # Filter into our window
                if start_dt <= ts_naive <= end_dt:
                    results.append(row)
            # If the oldest post on this page is already before start_dt,
            # then subsequent pages will be even older; we can stop.
            if page_oldest is not None and page_oldest < start_dt:
                break
            # Advance pagination; Substack uses offset-based paging
            offset += page_size
            time.sleep(SLEEP_S)
        except Exception:
            break
    return results

def even_monthly_sample_last_2y(df_in: pd.DataFrame, target: int, end_dt: pd.Timestamp) -> pd.DataFrame:
    """
    Stratified sampling: aim for an even distribution across the last 24 months [end_dt-23M .. end_dt].
    Fills quotas per month; redistributes leftovers; deduplicates by URL.
    """
    if df_in.empty or target <= 0:
        return df_in

    # Build last-24-month window
    end_dt = pd.Timestamp(end_dt)
    since_2y = end_dt - pd.DateOffset(months=24) + pd.DateOffset(days=1)
    df = df_in.copy()
    df["published"] = pd.to_datetime(df["published"], utc=True)
    df["published_naive"] = df["published"].dt.tz_convert(None)
    df = df[(df["published_naive"] >= since_2y) & (df["published_naive"] <= end_dt)].copy()
    if df.empty:
        return df_in.head(target)

    # Month labels
    df["month"] = df["published_naive"].dt.to_period("M").astype(str)
    all_months = pd.period_range(since_2y, end_dt, freq="M").astype(str).tolist()

    # Deduplicate by URL (keep most recent)
    df = df.sort_values("published", ascending=False).drop_duplicates("url", keep="first")

    base_quota = target // max(len(all_months), 1)
    remainder = target - base_quota * max(len(all_months), 1)

    picks = []
    leftover = 0
    rng_state = 42
    for m in all_months:
        grp = df[df["month"] == m]
        n_avail = len(grp)
        take = min(base_quota, n_avail)
        if take > 0:
            picks.append(grp.sample(n=take, random_state=rng_state))
        # accumulate leftover if not enough in this month
        leftover += (base_quota - take)

    chosen = pd.concat(picks) if picks else pd.DataFrame(columns=df.columns)
    chosen_urls = set(chosen["url"]) if not chosen.empty else set()

    # Distribute remainder + leftover across months with capacity
    need = target - len(chosen)
    if need > 0:
        # Candidate pool: remaining rows
        remaining = df[~df["url"].isin(chosen_urls)]
        # Prefer months with fewer selections so far
        month_counts = chosen["month"].value_counts().to_dict() if not chosen.empty else {}
        # Sort remaining by ascending month count, then random
        remaining = remaining.assign(_cnt=remaining["month"].map(lambda x: month_counts.get(x, 0)))
        # Stable random order with state
        remaining = remaining.sample(frac=1.0, random_state=rng_state).sort_values(["_cnt"]).drop(columns=["_cnt"])  
        chosen = pd.concat([chosen, remaining.head(need)])

    return chosen.drop(columns=[c for c in ["month", "published_naive"] if c in chosen.columns]).head(target)

# ---------- 1) Collect RSS metadata FIRST ----------
meta_rows = []
for pub_name, rss_url in SUBS.items():
    d = feedparser.parse(rss_url)
    if not d.entries:
        print(f"[warn] No entries for {pub_name} ({rss_url})")
        continue

    per_pub = []
    for e in d.entries:
        ts = safe_dt_from_entry(e)
        if ts is None:
            continue
        ts_naive = pd.Timestamp(ts).tz_convert(None)
        if not (START_DATE <= ts_naive <= END_DATE):
            continue
        title = getattr(e, "title", "").strip()
        link  = getattr(e, "link", "")
        summary_text = rss_content_to_text(e)
        per_pub.append({
            "pub": pub_name,
            "title": title,
            "url": link,
            "published": pd.Timestamp(ts),
            "rss_text": summary_text,
        })

    # Augment with Substack archive if needed to reach target
    if len(per_pub) < TARGET:
        try:
            extra = fetch_substack_archive(rss_url, START_DATE, END_DATE, TARGET)
            for row in extra:
                per_pub.append({
                    "pub": pub_name,
                    "title": row.get("title", ""),
                    "url": row.get("url", ""),
                    "published": pd.Timestamp(row.get("published")),
                    "rss_text": row.get("rss_text", ""),
                })
        except Exception:
            pass

    if not per_pub:
        print(f"[warn] No entries in range for {pub_name}")
        continue

    df_pub = pd.DataFrame(per_pub)
    # Deduplicate & enforce date window again
    df_pub["published"] = pd.to_datetime(df_pub["published"], utc=True)
    df_pub = df_pub.sort_values("published", ascending=False).drop_duplicates("url", keep="first")
    df_pub_naive = df_pub["published"].dt.tz_convert(None)
    df_pub = df_pub[(df_pub_naive >= START_DATE) & (df_pub_naive <= END_DATE)].reset_index(drop=True)

    # Even monthly sample across the last two years to exactly TARGET when possible
    end_dt = END_DATE
    df_pub_sampled = even_monthly_sample_last_2y(df_pub, TARGET, end_dt)

    meta_rows.append(df_pub_sampled)
    print(f"[ok] {pub_name}: collected={len(df_pub)}, sampled={len(df_pub_sampled)}")

df_meta = pd.concat(meta_rows, ignore_index=True) if meta_rows else pd.DataFrame(columns=["pub","title","url","published","rss_text"])
if df_meta.empty:
    raise SystemExit("No rows collected. Check feeds/date range.")
df_meta["published"] = pd.to_datetime(df_meta["published"], utc=True)
df_meta = df_meta.sort_values(["pub", "published"], ascending=[True, False]).reset_index(drop=True)
df_meta.to_parquet(META_PATH, index=False)
print(f"Wrote {META_PATH} ({len(df_meta)} rows)")

# ---------- 2) Clean/augment text ----------
clean_rows = []
for _, row in df_meta.iterrows():
    text = normalize_text(row.get("rss_text", "") or "")
    if symbol_zero_check(text) and isinstance(row["url"], str) and row["url"].startswith("http"):
        html = fetch_html(row["url"])
        page_text = normalize_text(html_to_text(html))
        # Prefer signal (emoji/dash) or meaningful length
        better = (DASH_RE.search(page_text) and not DASH_RE.search(text)) or (len(page_text) > len(text) * 1.25)
        if better:
            text = page_text

    clean_rows.append({
        "pub": row["pub"],
        "title": row["title"],
        "url": row["url"],
        "published": row["published"],
        "text": text,
        "n_chars": len(text),
    })

df_clean = pd.DataFrame(clean_rows)
df_clean.to_parquet(CLEAN_PATH, index=False)
print(f"Wrote {CLEAN_PATH} ({len(df_clean)} rows)")