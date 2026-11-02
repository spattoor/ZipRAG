"""Fetch 18 real neighborhood articles; no model calls or credentials needed."""
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
API = "https://en.wikipedia.org/w/api.php"
TITLES = [
    "Allandale, Austin, Texas", "Barton Hills, Austin, Texas",
    "Bouldin Creek, Austin, Texas", "Brentwood, Austin, Texas",
    "Bryker Woods, Austin, Texas", "Clarksville Historic District (Austin, Texas)",
    "Crestview, Austin, Texas", "Downtown Austin",
    "East Riverside-Oltorf, Austin, Texas", "Hyde Park (Austin, Texas)",
    "Mueller Community", "Northwest Hills, Austin, Texas",
    "Oak Hill, Austin, Texas", "Rosedale, Austin, Texas", "South Congress",
    "Tarrytown, Austin, Texas", "Travis Heights, Austin, Texas", "Zilker, Austin, Texas",
]


def fetch(title):
    params = dict(action="query", format="json", formatversion=2, redirects=1,
                  prop="extracts|info|revisions", explaintext=1, inprop="url",
                  rvprop="ids|timestamp", titles=title, maxlag=5)
    request = Request(API + "?" + urlencode(params), headers={
        "User-Agent": "ZipRAG/0.1 (https://github.com/spattoor; educational RAG demo)"})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=25) as response:
                data = json.loads(response.read(2_000_001))
            if "error" in data:
                raise ValueError(data["error"].get("code", "Wikipedia API error"))
            page = data["query"]["pages"][0]
            if page.get("missing") or len(page.get("extract", "").strip()) < 200:
                raise ValueError("Page missing or article too short")
            return page
        except (HTTPError, URLError, TimeoutError, ValueError) as error:
            if attempt == 2:
                raise RuntimeError(f"{title}: {error}") from error
            time.sleep(2 ** attempt)


def main():
    folder = ROOT / "docs"
    folder.mkdir(exist_ok=True)
    records, failures = [], []
    for title in TITLES:
        try:
            page = fetch(title)
            filename = f"{page['pageid']}.txt"
            text = page["extract"].strip()
            (folder / filename).write_text(text + "\n", encoding="utf-8")
            records.append(dict(file=filename, title=page["title"], url=page["fullurl"],
                revision=page["revisions"][0]["revid"],
                retrieved_at=datetime.now(timezone.utc).isoformat(),
                sha256=hashlib.sha256((text + "\n").encode()).hexdigest(),
                license="CC BY-SA 4.0; see article history for contributors"))
            print(f"Saved {len(records):02d}: {page['title']}", flush=True)
        except RuntimeError as error:
            failures.append(str(error))
            print(f"FAILED: {error}", flush=True)
    (folder / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    print(f"Downloaded {len(records)}/{len(TITLES)} documents; {len(failures)} failures.")
    if failures:
        raise SystemExit("Some downloads failed. Resolve these before Step 2.")


if __name__ == "__main__":
    main()
