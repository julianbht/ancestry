"""Fetch public-domain Curie / Joliot-Curie photos from Wikimedia Commons.

Reproducible sourcing for the quickstart dummy dataset (see docs/going-public.md).
For each (deceased) person in the quickstart family tree it queries the Commons
API, picks one or more freely-licensed images (public domain / CC0 / "no
restrictions", or CC-BY with attribution), downloads web-sized copies into
quickstart/data/raw/album/, and writes CREDITS.json linking every file back to
its Commons page, author, and licence. Only the Commons API + stdlib are used.

Living people are intentionally excluded (Hélène Langevin-Joliot I0005 and Pierre
Joliot I0006) — the demo ships no images of the living. Everyone else here died
long ago and has public-domain photographs.

    uv run python scripts/fetch_quickstart_photos.py            # fetch missing
    uv run python scripts/fetch_quickstart_photos.py --force    # re-fetch all

Can't see the images, so selection is title/licence/author heuristics; the chosen
and rejected candidates are printed. If a pick is poor, pin an exact Commons file
in TARGETS (a ``files`` entry) and re-run with --force.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from pipeline.shared.paths import PROJECT_ROOT

API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = (
    "ancestry-photo-pipeline-quickstart/0.3 "
    "(reproducible public-domain dummy-data fetch; non-commercial; "
    "https://commons.wikimedia.org) Python-urllib"
)
ALBUM = PROJECT_ROOT / "quickstart" / "data" / "raw" / "album"
THUMB_WIDTH = 1200   # download a web-sized copy, not the full scan
MIN_WIDTH = 450      # skip thumbnails too small to detect a face in

# person_ids reference the quickstart family tree (family-tree-data.csv). Each
# target resolves to Commons file(s) either directly (``files``: exact titles) or
# by search (``search`` + ``require`` name token, up to ``count`` distinct hits).
# I0005/I0006 (the Joliot children) are omitted on purpose — still living.
TARGETS = [
    # Group / multi-person shots (both Pierre & Marie) — welcome in the demo.
    {"slug": "group-pierre-marie", "person_ids": ["I0000", "I0001"],
     "files": ["File:Pierre and Marie Curie.jpg"]},
    {"slug": "group-pierre-marie-lab", "person_ids": ["I0000", "I0001"],
     "files": ["File:Marie and Pierre Curie Converse.jpg"]},
    # Individuals — pinned public-domain portraits (validated), several per person
    # where available for age variety across face-recognition examples.
    {"slug": "pierre-curie", "person_ids": ["I0000"],
     "files": ["File:Pierre Curie by Dujardin c1906.jpg"]},
    {"slug": "marie-curie", "person_ids": ["I0001"], "files": [
        "File:Marie Curie, portrait, 1900.jpg",
        "File:Marie Curie c. 1920s.jpg",
        "File:Mariecurie.jpg",
    ]},
    {"slug": "irene-joliot-curie", "person_ids": ["I0002"], "files": [
        "File:Irène Joliot-Curie 1936.jpg",
        "File:Joliot-curie.jpg",
    ]},
    {"slug": "frederic-joliot-curie", "person_ids": ["I0003"], "files": [
        "File:Joliot-Curie Harcourt 1948 3.jpg",
    ]},
    {"slug": "eve-curie", "person_ids": ["I0004"], "files": ["File:Ève Curie (1943).jpg"]},

    # --- Extra images (user-provided Commons links) -------------------------
    # Group / multi-person shots — welcome; unlisted faces (sisters, colleagues,
    # Solvay attendees, Queen Mary…) are simply the demo's "unknown" people.
    {"slug": "group-marie-pierre-irene", "person_ids": ["I0000", "I0001", "I0002"],
     "files": ["File:Marie Pierre Irene Curie.jpg"]},
    {"slug": "group-pierre-marie-1895", "person_ids": ["I0000", "I0001"],
     "files": ["File:Pierre Curie et Marie Sklodowska Curie 1895.jpg"]},
    {"slug": "group-pierre-marie-at-work", "person_ids": ["I0000", "I0001"],
     "files": ["File:Pierre and Marie Curie at work.jpg"]},
    {"slug": "group-irene-frederic-1935", "person_ids": ["I0002", "I0003"],
     "files": ["File:Irène et Frédéric Joliot-Curie 1935.jpg"]},
    {"slug": "group-solvay-1927", "person_ids": ["I0001"],
     "files": ["File:Solvay conference 1927 Version2.jpg"]},
    {"slug": "group-eve-1937", "person_ids": ["I0004"],
     "files": ["File:Ève Curie, Henri Coutard, Queen Mary, Hilda Runciman 1937.jpg"]},
    # Władysław Skłodowski (I0009) — solo, plus the family group with daughter Marie.
    {"slug": "wladyslaw-sklodowski", "person_ids": ["I0009"],
     "files": ["File:Ś. p. Władysław Skłodowski (55681).jpg"]},
    {"slug": "group-sklodowski-family", "person_ids": ["I0009", "I0001"],
     "files": ["File:Sklodowski Family Wladyslaw and his daughters Maria Bronislawa Helena.jpg"]},
    # The "Bronisława" here is Marie's SISTER, not her mother I0010 — tag Marie only.
    {"slug": "marie-and-sister-1886", "person_ids": ["I0001"],
     "files": ["File:Maria Sklodowska et sa sœur Bronislawa en 1886.jpg"]},
    # More individual portraits, for age variety.
    {"slug": "irene-joliot-curie-1921", "person_ids": ["I0002"],
     "files": ["File:Irène Joliot-Curie (1897-1956), 1921.jpg"]},
    {"slug": "irene-joliot-curie-c1935", "person_ids": ["I0002"],
     "files": ["File:Irène Joliot-Curie (1897-1956), c. 1935.jpg"]},
    {"slug": "frederic-joliot-curie-harcourt", "person_ids": ["I0003"], "files": [
        "File:Frédéric Joliot-Curie Harcourt.jpg",
        "File:Joliot-Curie Harcourt 1948.jpg",
        "File:Joliot-Curie Harcourt 1948 2.jpg",
    ]},
    {"slug": "eve-curie-1937", "person_ids": ["I0004"], "files": ["File:Ève Curie 1937.jpg"]},
    {"slug": "eve-curie-1939", "person_ids": ["I0004"], "files": ["File:Ève Curie (-1939).jpg"]},
    # Skipped on purpose: "Władysław Józef Skłodowski … 1926" is a different, later
    # man (Marie's father died 1902). Still unfilled: I0007 (Eugène Curie) and
    # I0010 (the mother Bronisława). Living relatives I0005/I0006 omitted.
]

# Title fragments that signal "not a portrait photo" — filtered out of search.
_BAD_TITLE = re.compile(
    r"stamp|postage|monument|plaque|grave|tomb|signature|logo|medal|coin|banknote"
    r"|bust|statue|memorial|diagram|\bmap\b|document|letter|envelope|crater|asteroid"
    r"|street|square|museum|building|university|institut|poster|caricature|comic"
    r"|book|award|prize|franc|bill\b|nobel|diplom|drawing|painting|engraving"
    r"|illustration|sketch|artwork|watercolo",
    re.IGNORECASE,
)
# Authors that mean the file is a stamp/postal issue rather than a photograph.
_BAD_AUTHOR = re.compile(r"post der|deutsche post|bundespost|postage|poczta|correos|posta|pošta", re.IGNORECASE)

_last_request = [0.0]


def _read(req: urllib.request.Request, timeout: int) -> bytes:
    """Fetch bytes, throttled and with backoff on Commons rate limits (HTTP 429)."""
    for attempt in range(4):
        gap = time.monotonic() - _last_request[0]
        if gap < 2.5:  # be polite: keep requests serial and well spaced
            time.sleep(2.5 - gap)
        _last_request[0] = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 3:
                time.sleep(5 * (attempt + 1))
                continue
            raise
    raise RuntimeError("unreachable")


def _api(params: dict) -> dict:
    params = {**params, "format": "json", "formatversion": "2"}
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return json.loads(_read(req, timeout=30))


def _strip_html(value: str | None) -> str:
    text = re.sub(r"<[^>]+>", "", value or "").strip()
    # Commons sometimes returns the Artist string doubled ("XX"); collapse it.
    half = len(text) // 2
    if text and len(text) % 2 == 0 and text[:half] == text[half:]:
        text = text[:half]
    return text


def _clean_date(value: str | None) -> str:
    # extmetadata dates look like "circa 1920date QS:P,+1920-..." — keep the prose.
    return re.split(r"date QS:", _strip_html(value))[0].strip()


def _license(extmeta: dict) -> str:
    return (extmeta.get("LicenseShortName", {}).get("value") or "unknown").strip()


def _is_free(lic: str) -> bool:
    """True for reuse-OK licences: public domain, CC0, "no restrictions", or CC-BY(-SA).
    No living people are fetched, so old CC-BY archival scans are fine — the required
    attribution (author + licence) is recorded in CREDITS.json."""
    low = lic.lower()
    return ("public domain" in low or low.startswith("pd") or low == "cc0"
            or "no restriction" in low or "cc by" in low or "cc-by" in low)


def _imageinfo(title: str) -> dict | None:
    data = _api({
        "action": "query", "titles": title, "prop": "imageinfo",
        "iiprop": "url|extmetadata|size|mime", "iiurlwidth": THUMB_WIDTH,
    })
    pages = data.get("query", {}).get("pages", [])
    if not pages or "missing" in pages[0] or not pages[0].get("imageinfo"):
        return None
    return pages[0]["imageinfo"][0]


def _pick_candidates(term: str, require: str, count: int) -> list[tuple[str, dict]]:
    """Up to `count` distinct public-domain portrait-ish files, in search order."""
    data = _api({
        "action": "query", "generator": "search", "gsrsearch": term,
        "gsrnamespace": "6", "gsrlimit": "40", "prop": "imageinfo",
        "iiprop": "url|extmetadata|size|mime", "iiurlwidth": THUMB_WIDTH,
    })
    pages = sorted(data.get("query", {}).get("pages", []), key=lambda p: p.get("index", 0))
    chosen: list[tuple[str, dict]] = []
    rejected: list[str] = []
    for page in pages:
        title = page.get("title", "")
        info = (page.get("imageinfo") or [None])[0]
        if not info:
            continue
        lic = _license(info.get("extmetadata", {}))
        author = _strip_html(info.get("extmetadata", {}).get("Artist", {}).get("value"))
        reason = None
        if info.get("mime") not in ("image/jpeg", "image/png"):
            reason = f"mime={info.get('mime')}"
        elif require.lower() not in title.lower():
            reason = "name-mismatch"
        elif _BAD_TITLE.search(title):
            reason = "non-portrait"
        elif _BAD_AUTHOR.search(author):
            reason = "stamp/postal"
        elif (info.get("width") or 0) < MIN_WIDTH:
            reason = f"small({info.get('width')}px)"
        elif not _is_free(lic):
            reason = f"non-free({lic})"
        if reason:
            rejected.append(f"{title} [{reason}]")
            continue
        chosen.append((title, info))
        if len(chosen) >= count:
            break
    if rejected:
        print("      rejected:", "; ".join(rejected[:4]))
    return chosen


def _download(url: str, dest: Path) -> int:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    blob = _read(req, timeout=60)
    dest.write_bytes(blob)
    return len(blob)


def _credit(filename: str, person_ids: list[str], title: str, info: dict) -> dict:
    em = info.get("extmetadata", {})
    return {
        "file": filename,
        "person_ids": person_ids,
        "title": title,
        "commons_page": info.get("descriptionurl"),
        "source_url": info.get("url"),
        "downloaded_from": info.get("thumburl", info.get("url")),
        "original_size": [info.get("width"), info.get("height")],
        "author": _strip_html(em.get("Artist", {}).get("value")) or "Unknown",
        "license": _license(em),
        "date": _clean_date(em.get("DateTimeOriginal", {}).get("value")),
    }


def _resolve(target: dict) -> list[tuple[str, dict]]:
    if "files" in target:
        out = []
        for title in target["files"]:
            info = _imageinfo(title)
            if info and _is_free(_license(info.get("extmetadata", {}))):
                out.append((title, info))
            else:
                print(f"      SKIP {title}: missing or non-free")
        return out
    return _pick_candidates(target["search"], target["require"], target.get("count", 1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="re-fetch even if the file exists")
    args = ap.parse_args()

    ALBUM.mkdir(parents=True, exist_ok=True)
    credits: list[dict] = []
    for t in TARGETS:
        slug, person_ids = t["slug"], t["person_ids"]
        print(f"- {slug} ({', '.join(person_ids)})")
        picked = _resolve(t)
        if not picked:
            print("      no usable public-domain image found — skipping")
            continue

        multi = len(picked) > 1 or t.get("count", 1) > 1
        for i, (title, info) in enumerate(picked, 1):
            filename = f"{slug}-{i}.jpg" if multi else f"{slug}.jpg"
            dest = ALBUM / filename
            entry = _credit(filename, person_ids, title, info)
            print(f"      {filename} <- {title}")
            print(f"        {entry['license']} | {entry['author']} | "
                  f"{entry['original_size'][0]}x{entry['original_size'][1]} | {entry['date'] or 'n.d.'}")
            if dest.exists() and not args.force:
                print("        exists — keeping (use --force to replace)")
            else:
                try:
                    size = _download(entry["downloaded_from"], dest)
                    print(f"        downloaded {size / 1e3:.0f} KB")
                except (urllib.error.HTTPError, urllib.error.URLError) as e:
                    print(f"        FAILED ({e}) — re-run to retry this one")
                    continue  # don't credit a file we didn't get
            credits.append(entry)

    if not credits:
        sys.exit("No images fetched.")
    creditsfile = ALBUM / "CREDITS.json"
    creditsfile.write_text(json.dumps({
        "note": "Freely-licensed images from Wikimedia Commons, used as quickstart "
                "dummy data. Most are public domain; CC-BY files require the recorded "
                "author + licence as attribution. See each commons_page for full terms. "
                "No images of living people are included.",
        "images": credits,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nWrote {len(credits)} image(s) + {creditsfile.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
