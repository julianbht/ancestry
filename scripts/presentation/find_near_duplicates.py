"""Find near-duplicate photos: the same physical print photographed more than once.

The bag of prints contained copies — the same picture printed several times —
and every print was photographed, so the dataset holds semantic duplicates that
are NOT pixel-identical (different angle, lighting, framing). Exact-hash dedup
(see scripts/data_quality.py) misses these by construction.

Detection is two stages, run on the frame_crop outputs (already deframed to just
the print, which removes most background/translation variance):

  1. Candidate generation — a 64-bit perceptual hash (pHash) per print. Pairs
     whose Hamming distance is small are cheap to enumerate over all ~1.7k
     images and form the candidate set. Loose threshold = high recall.

  2. Geometric verification — ORB keypoints + a RANSAC homography between each
     candidate pair. Two photos of the *same flat print* are related by a real
     planar homography, so a high RANSAC inlier count is strong evidence they
     are the same print; unrelated photos that happen to hash similarly produce
     few inliers and are rejected. This is what makes it robust to camera angle.

Confirmed pairs are unioned into duplicate groups. A montage per group is written
to data/debug/near_duplicates/ so the result can be eyeballed.

Usage:
    uv run python scripts/find_near_duplicates.py
    uv run python scripts/find_near_duplicates.py --hash-threshold 26 --min-inliers 15
    uv run python scripts/find_near_duplicates.py --no-montages
"""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
FRAME_CROP_DIR = PROJECT_ROOT / "data" / "steps" / "frame_crop"
OUT_DIR = PROJECT_ROOT / "data" / "debug" / "near_duplicates"


# --------------------------------------------------------------------------- #
# Stage 1: perceptual hash (pHash)
# --------------------------------------------------------------------------- #
def phash(image_bgr: np.ndarray, hash_size: int = 8) -> int:
    """64-bit DCT-based perceptual hash, robust to scale/lighting/mild warp."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (hash_size * 4, hash_size * 4), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(small.astype(np.float32))
    low = dct[:hash_size, :hash_size]
    med = np.median(low[1:, 1:])  # exclude DC term from the threshold
    bits = (low > med).flatten()
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return value


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


# --------------------------------------------------------------------------- #
# Stage 2: ORB + RANSAC geometric verification
# --------------------------------------------------------------------------- #
def orb_features(image_bgr: np.ndarray, n_features: int, max_side: int):
    """Detect ORB keypoints/descriptors on a size-normalised grayscale image."""
    h, w = image_bgr.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        image_bgr = cv2.resize(image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=n_features)
    kps, desc = orb.detectAndCompute(gray, None)
    pts = np.float32([kp.pt for kp in kps]) if kps else np.empty((0, 2), np.float32)
    return pts, desc


def homography_inliers(feat_a, feat_b, ratio: float) -> int:
    """Count RANSAC homography inliers between two ORB feature sets."""
    (pts_a, desc_a), (pts_b, desc_b) = feat_a, feat_b
    if desc_a is None or desc_b is None or len(desc_a) < 4 or len(desc_b) < 4:
        return 0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw = matcher.knnMatch(desc_a, desc_b, k=2)
    good = [m for pair in raw if len(pair) == 2 for m, n in [pair] if m.distance < ratio * n.distance]
    if len(good) < 8:
        return 0

    src = np.float32([pts_a[m.queryIdx] for m in good]).reshape(-1, 1, 2)
    dst = np.float32([pts_b[m.trainIdx] for m in good]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return int(mask.sum()) if mask is not None else 0


# --------------------------------------------------------------------------- #
# Union-find for grouping confirmed pairs
# --------------------------------------------------------------------------- #
class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


# --------------------------------------------------------------------------- #
def write_montage(paths: list[Path], out_path: Path, thumb: int = 320) -> None:
    thumbs = []
    for p in paths:
        im = cv2.imread(str(p))
        if im is None:
            continue
        h, w = im.shape[:2]
        scale = thumb / max(h, w)
        im = cv2.resize(im, (int(w * scale), int(h * scale)))
        canvas = np.full((thumb, thumb, 3), 30, np.uint8)
        yh, xw = im.shape[:2]
        canvas[(thumb - yh) // 2 : (thumb - yh) // 2 + yh, (thumb - xw) // 2 : (thumb - xw) // 2 + xw] = im
        thumbs.append(canvas)
    if thumbs:
        cv2.imwrite(str(out_path), cv2.hconcat(thumbs))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hash-threshold", type=int, default=24, help="max pHash Hamming distance for a candidate pair (default 24/64)")
    parser.add_argument("--top-k", type=int, default=6, help="keep each image's K nearest neighbours as candidates (default 6)")
    parser.add_argument("--min-inliers", type=int, default=18, help="min RANSAC homography inliers to confirm a duplicate (default 18)")
    parser.add_argument("--orb-features", type=int, default=1500, help="ORB features per image (default 1500)")
    parser.add_argument("--max-side", type=int, default=800, help="downscale long edge to this before ORB (default 800)")
    parser.add_argument("--ratio", type=float, default=0.75, help="Lowe ratio-test threshold (default 0.75)")
    parser.add_argument("--no-montages", action="store_true", help="do not write per-group montage images")
    parser.add_argument("--out", type=Path, help="also write a Markdown report to this path (e.g. presentation/near_duplicates.md)")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N images (smoke test)")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

    paths = sorted(FRAME_CROP_DIR.rglob("*.jpg"))
    if args.limit:
        paths = paths[: args.limit]
    n = len(paths)
    print(f"Loaded {n} frame_crop images")

    # --- Stage 1: decode each image once, compute pHash + ORB features together ---
    print("Stage 1: computing perceptual hashes + ORB features ...")
    hashes: list[int] = []
    feats: dict[int, tuple] = {}
    for idx, p in enumerate(paths):
        im = cv2.imread(str(p))
        if im is None:
            hashes.append(0)
            feats[idx] = (np.empty((0, 2), np.float32), None)
            continue
        hashes.append(phash(im))
        feats[idx] = orb_features(im, args.orb_features, args.max_side)

    # Candidate pairs = each image's K nearest neighbours by Hamming distance,
    # capped at hash_threshold. A fixed radius alone explodes on this dataset
    # (old B&W prints cluster tightly in hash space); top-K keeps the candidate
    # set bounded at ~K*n while still pairing true copies (mutual neighbours).
    candidate_set: set[tuple[int, int]] = set()
    for i in range(n):
        dists = sorted(
            ((hamming(hashes[i], hashes[j]), j) for j in range(n) if j != i),
            key=lambda t: t[0],
        )[: args.top_k]
        for d, j in dists:
            if d <= args.hash_threshold:
                candidate_set.add((min(i, j), max(i, j)))
    candidates = sorted(candidate_set)
    print(f"Stage 1: {len(candidates):,} candidate pairs (top-{args.top_k} neighbours, Hamming ≤ {args.hash_threshold})")

    # --- Stage 2: geometric verification of candidate pairs ---
    print("Stage 2: geometric verification (ORB + RANSAC) ...")
    uf = UnionFind(n)
    confirmed: list[tuple[int, int, int]] = []
    for i, j in candidates:
        inliers = homography_inliers(feats[i], feats[j], args.ratio)
        if inliers >= args.min_inliers:
            uf.union(i, j)
            confirmed.append((i, j, inliers))
    print(f"Stage 2: {len(confirmed):,} confirmed duplicate pairs (≥ {args.min_inliers} inliers)")

    # --- Group ---
    groups: dict[int, list[int]] = {}
    for idx in range(n):
        groups.setdefault(uf.find(idx), []).append(idx)
    dup_groups = sorted((m for m in groups.values() if len(m) > 1), key=len, reverse=True)

    extra = sum(len(g) - 1 for g in dup_groups)
    print()
    print("=" * 70)
    print(f"Near-duplicate groups: {len(dup_groups)}")
    print(f"Images involved:       {sum(len(g) for g in dup_groups)}")
    print(f"Redundant copies:      {extra}  ({extra / n:.1%} of {n} prints)")
    print("=" * 70)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = []
    for gi, members in enumerate(dup_groups):
        rels = [str(paths[m].relative_to(PROJECT_ROOT)) for m in members]
        report.append({"group": gi, "size": len(members), "members": rels})
        print(f"\n[group {gi}] {len(members)} prints:")
        for r in rels:
            print(f"    {r}")
        if not args.no_montages:
            write_montage([paths[m] for m in members], OUT_DIR / f"group_{gi:03d}_n{len(members)}.png")

    (OUT_DIR / "near_duplicates.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport: {(OUT_DIR / 'near_duplicates.json').relative_to(PROJECT_ROOT)}")
    if not args.no_montages:
        print(f"Montages: {OUT_DIR.relative_to(PROJECT_ROOT)}/group_*.png")

    if args.out:
        out = args.out if args.out.is_absolute() else PROJECT_ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_markdown(report, n, extra, args.min_inliers, args.hash_threshold), encoding="utf-8")
        print(f"Markdown report written to {out.relative_to(PROJECT_ROOT)}")


def render_markdown(report: list[dict], n: int, extra: int, min_inliers: int, hash_threshold: int) -> str:
    """Markdown summary of the near-duplicate groups, mirroring data_quality.py."""
    out: list[str] = ["# Near-Duplicate Report — ancestry photo dataset", ""]
    out.append(
        "*Semantic duplicates: the same physical print photographed more than once "
        "(different angle/lighting/framing), so they are not pixel-identical and are "
        "missed by exact-hash dedup.*"
    )
    out.append("")
    out.append(
        f"Method: perceptual-hash candidates (Hamming ≤ {hash_threshold}/64) confirmed by "
        f"ORB + RANSAC homography (≥ {min_inliers} inliers), run on the deframed `frame_crop` prints."
    )
    out.append("")
    out.append("| Metric | Count | Of total | % |")
    out.append("| --- | ---: | ---: | ---: |")
    images_involved = sum(g["size"] for g in report)
    out.append(f"| prints analysed | {n:,} |  |  |")
    out.append(f"| near-duplicate groups | {len(report):,} |  |  |")
    out.append(f"| prints involved in a group | {images_involved:,} | {n:,} | {images_involved / n:.1%} |")
    out.append(f"| redundant copies (group size − 1) | {extra:,} | {n:,} | {extra / n:.1%} |")
    out.append("")
    if report:
        out.append("## Groups")
        out.append("")
        out.append("| Group | Size | Members |")
        out.append("| ---: | ---: | --- |")
        for g in report:
            members = "<br>".join(g["members"])
            out.append(f"| {g['group']} | {g['size']} | {members} |")
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    main()
