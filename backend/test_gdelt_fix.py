"""Quick diagnostic: verify _is_credible fix and sample GDELT output."""
import sys, time, requests
sys.path.insert(0, ".")
from app.ml.corpus.rss_fetcher import CALABARZON_FOOD_SIGNALS, _is_credible

# --- 1. Confirm _is_credible works with full URLs ---
print("=== _is_credible sanity check ===")
tests = [
    ("https://www.rappler.com/article/123", True),
    ("https://newsinfo.inquirer.net/article", True),
    ("https://pna.gov.ph/article", True),
    ("https://manilatimes.net/article", True),
    ("https://bworldonline.com/article", True),
    ("https://tribune.net.ph/article", False),
    ("https://ttnworldwide.com/article", False),
]
for url, expected in tests:
    result = _is_credible(url)
    status = "OK" if result == expected else "FAIL"
    print(f"  [{status}] _is_credible('{url[:40]}...') = {result}  (expected {expected})")

# --- 2. Live GDELT test ---
print("\n=== Live GDELT test (waiting 3s for rate limit) ===")
time.sleep(3)
url = (
    "https://api.gdeltproject.org/api/v2/doc/doc"
    "?query=%22Batangas%22%20(food%20OR%20hunger%20OR%20rice%20OR%20relief)"
    "&mode=ArtList&maxrecords=25"
    "&startdatetime=20200101000000&enddatetime=20200401000000"
    "&sort=DateDesc&format=json&sourcecountry=PH"
)
r = requests.get(url, timeout=30)
print("HTTP status:", r.status_code, "| Body length:", len(r.text))

if r.status_code == 200 and r.text.strip():
    articles = r.json().get("articles") or []
    print(f"Raw articles from GDELT: {len(articles)}")
    passed = 0
    for a in articles:
        full_url = a.get("url", "")
        title    = a.get("title", "")
        dom      = a.get("domain", "")
        cred     = _is_credible(full_url)
        food     = any(kw in title.lower() for kw in CALABARZON_FOOD_SIGNALS)
        if cred and food:
            status = "PASS    "
            passed += 1
        elif not cred:
            status = "bad_dom "
        else:
            status = "no_food "
        print(f"  [{status}] {dom:30s} | {title[:55]}")
    print(f"\n--- {passed}/{len(articles)} articles pass both filters ---")
else:
    print("Empty or error response — GDELT may still be rate-limiting. Try again in 30s.")
