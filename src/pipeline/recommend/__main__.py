"""Face recommender CLI: "given a person, recommend other photos of them".

A content-based recommender over the InsightFace embeddings face_recognition
already produced. Compares two retrieval strategies (max-similarity vs centroid)
and evaluates them with Precision@k / Recall@k by two methods (automatic against
the ground truth, and manual over the full collection).

Modes:
    evaluate      Automatic Precision@k/Recall@k on the labelled pool, both
                  strategies, for one person and a macro-average over everyone
                  with enough labelled faces.
    retrieve      Rank the whole collection for one person with both strategies
                  and write a CSV of the top-k (with a blank `correct` column)
                  for manual review.
    score-manual  Read a reviewed CSV back and report manual Precision@k.

Examples:
    uv run python -m pipeline.recommend evaluate --person-id I0000
    uv run python -m pipeline.recommend retrieve --person-id I0000 --top-k 20 --out recommend_review.csv
    uv run python -m pipeline.recommend score-manual --in recommend_review.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from pipeline.recommend.index import FaceIndex, build_index
from pipeline.recommend.metrics import evaluate_person, ndcg_at_k
from pipeline.recommend.retrieval import STRATEGIES, recommend
from pipeline.shared.paths import DEBUG_DIR, PROJECT_ROOT

DEFAULT_KS = [1, 3, 5, 10]
DEFAULT_PERSON = "I0000"  # most-labelled person (Renate Käthe Maria Boldt, 44 faces)
MONTAGE_DIR = DEBUG_DIR / "recommend"  # regenerable review aids (gitignored data/)
REVIEWS_DIR = Path(__file__).parent / "reviews"  # hand-marked CSVs — precious, kept in-tree


def _names() -> dict[str, str]:
    try:
        from pipeline.gramps import load_family_tree

        return {p.id: p.full_name for p in load_family_tree()}
    except Exception:
        return {}


def _print_eval_table(index: FaceIndex, person_id: str, ks: list[int], names: dict[str, str]) -> None:
    label = names.get(person_id, person_id)
    n = len(index.indices_for_person(person_id))
    print(f"\nPerson {person_id} ({label}) — {n} labelled faces")
    print(f"  {'strategy':<10} " + "  ".join(f"P@{k:<3} R@{k:<3}" for k in ks))
    for strat in STRATEGIES:
        r = evaluate_person(index, person_id, strat, ks)
        cells = "  ".join(f"{r.precision[k]:.2f}  {r.recall[k]:.2f}" for k in ks)
        print(f"  {strat:<10} {cells}")


def _macro_average(index: FaceIndex, ks: list[int], min_faces: int) -> None:
    eligible = [p for p, c in index.person_counts().items() if c >= min_faces]
    print(f"\nMacro-average over {len(eligible)} people with ≥{min_faces} labelled faces")
    print(f"  {'strategy':<10} " + "  ".join(f"P@{k:<3} R@{k:<3}" for k in ks))
    for strat in STRATEGIES:
        results = [evaluate_person(index, p, strat, ks) for p in eligible]
        prec = {k: sum(r.precision[k] for r in results) / len(results) for k in ks}
        rec = {k: sum(r.recall[k] for r in results) / len(results) for k in ks}
        cells = "  ".join(f"{prec[k]:.2f}  {rec[k]:.2f}" for k in ks)
        print(f"  {strat:<10} {cells}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    index = build_index()
    names = _names()
    _print_eval_table(index, args.person_id, DEFAULT_KS, names)
    _macro_average(index, DEFAULT_KS, args.min_faces)


def _parse_ranks(spec: str | None, top_k: int) -> tuple[int, int]:
    """'66-85' -> (66, 85); '66' -> (66, 66); None -> (1, top_k). 1-indexed, inclusive."""
    if not spec:
        return 1, top_k
    lo_s, _, hi_s = spec.partition("-")
    lo = int(lo_s)
    hi = int(hi_s) if hi_s else lo
    if lo < 1 or hi < lo:
        sys.exit(f"bad --ranks {spec!r}; use e.g. 66-85")
    return lo, min(hi, top_k)


def cmd_retrieve(args: argparse.Namespace) -> None:
    # One CSV per method (paired 1:1 with that method's montage), named with
    # person + method + k so runs never collide. An existing CSV is kept as-is
    # (its marks are precious) and only the missing method(s) are written —
    # unless --force regenerates everything. --montage-only skips CSVs entirely
    # and just (re)writes the review contact sheets.
    write_csv = not args.montage_only
    make_montage = args.montage or args.montage_only
    rank_lo, rank_hi = _parse_ranks(args.ranks, args.top_k)
    rank_suffix = f"_r{rank_lo}-{rank_hi}" if args.ranks else ""

    out_dir = args.out_dir or REVIEWS_DIR
    out_dir = out_dir if out_dir.is_absolute() else Path.cwd() / out_dir
    csv_path = {s: out_dir / f"recommend_{args.person_id}_{s}_top{args.top_k}.csv" for s in STRATEGIES}

    if write_csv:
        to_process = [s for s in STRATEGIES if args.force or not csv_path[s].exists()]
        for s in STRATEGIES:
            if s not in to_process:
                print(f"  {s:<10} kept (already exists: {csv_path[s].name})")
        if not to_process:
            print("Nothing to write — all CSVs already exist (use --force to regenerate).")
            return
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        to_process = list(STRATEGIES)

    index = build_index()
    names = _names()
    query = index.indices_for_person(args.person_id)
    if not query:
        sys.exit(f"No labelled faces for {args.person_id}")
    print(f"Querying with {len(query)} reference faces of {args.person_id} ({names.get(args.person_id, '?')})")

    fields = ["method", "rank", "score", "source_photo", "crop_image", "gt_person_id", "correct"]
    for strat in to_process:
        ranked = [(rank, idx, score) for rank, (idx, score) in enumerate(recommend(index, query, strat, args.top_k), start=1)]

        if write_csv:
            rows = []
            for rank, idx, score in ranked:
                gt = index.person_ids[idx]
                # Pre-fill 'correct' where ground truth already knows the answer.
                correct = "" if gt is None else ("1" if gt == args.person_id else "0")
                rows.append({
                    "method": strat,
                    "rank": rank,
                    "score": f"{score:.4f}",
                    "source_photo": index.source_photos[idx],
                    "crop_image": index.crop_images[idx],
                    "gt_person_id": gt or "",
                    "correct": correct,
                })
            with csv_path[strat].open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            n_blank = sum(1 for r in rows if r["correct"] == "")
            print(f"  {strat:<10} {len(rows)} rows -> {csv_path[strat].name}  ({n_blank} to mark)")

        if make_montage:
            selected = [(r, idx, sc) for r, idx, sc in ranked if rank_lo <= r <= rank_hi]
            paths = [(PROJECT_ROOT / index.crop_images[idx], f"#{r} {sc:.2f}") for r, idx, sc in selected]
            _write_montage(paths, MONTAGE_DIR / f"{args.person_id}_{strat}_top{args.top_k}{rank_suffix}.png")

    if write_csv:
        print(f"CSVs in {out_dir}; fill each 'correct' column (1/0), then run score-manual --in <both csvs>.")
    if make_montage:
        print(f"Montages: {MONTAGE_DIR.relative_to(PROJECT_ROOT)}/{args.person_id}_<method>_top{args.top_k}{rank_suffix}.png")


def _write_montage(items: list[tuple[Path, str]], out_path: Path, thumb: int = 256, cols: int = 5) -> None:
    """Grid of rank-labelled face thumbnails, as a review aid (not auto-opened)."""
    import cv2
    import numpy as np

    cells = []
    for path, caption in items:
        im = cv2.imread(str(path))
        canvas = np.full((thumb, thumb, 3), 30, np.uint8)
        if im is not None:
            h, w = im.shape[:2]
            scale = (thumb - 24) / max(h, w)
            im = cv2.resize(im, (int(w * scale), int(h * scale)))
            yh, xw = im.shape[:2]
            y0, x0 = (thumb - 24 - yh) // 2, (thumb - xw) // 2
            canvas[y0 : y0 + yh, x0 : x0 + xw] = im
        cv2.putText(canvas, caption, (6, thumb - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cells.append(canvas)

    rows = []
    for i in range(0, len(cells), cols):
        row = cells[i : i + cols]
        while len(row) < cols:
            row.append(np.full((thumb, thumb, 3), 30, np.uint8))
        rows.append(cv2.hconcat(row))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.vconcat(rows))


def cmd_score_manual(args: argparse.Namespace) -> None:
    rows = []
    for path in args.inp:
        rows.extend(csv.DictReader(path.open(encoding="utf-8")))
    # Preserve the order the methods first appear (i.e. the order of --in files),
    # so table rows line up with the listed source files.
    methods = list(dict.fromkeys(r["method"] for r in rows))

    # Evaluate at every standard cutoff that the reviewed depth supports.
    depth = min(len([r for r in rows if r["method"] == m]) for m in methods)
    ks = [k for k in (10, 25, 50, 100) if k <= depth] or [depth]

    # Per method: {k: (precision, ndcg)} or None if its top-k has blank marks.
    results: dict[str, dict[int, tuple[float, float]] | None] = {}
    for method in methods:
        ranked = sorted((r for r in rows if r["method"] == method), key=lambda r: int(r["rank"]))
        marks = [r["correct"].strip() for r in ranked]
        if any(m == "" for m in marks[: max(ks)]):
            results[method] = None
            continue
        hits = [m == "1" for m in marks]
        results[method] = {k: (sum(hits[:k]) / k, ndcg_at_k(hits, k)) for k in ks}

    print(f"Manual Precision@k / nDCG@k from {len(args.inp)} file(s) (reviewed depth {depth})")
    print(f"  {'method':<10} " + "  ".join(f"P@{k:<3} nDCG@{k:<3}" for k in ks))
    for method, res in results.items():
        if res is None:
            print(f"  {method:<10} (incomplete — some 'correct' cells in the top-{max(ks)} are blank)")
        else:
            print(f"  {method:<10} " + "  ".join(f"{res[k][0]:.2f}  {res[k][1]:.2f}" for k in ks))

    out = args.out or args.inp[0].parent / f"scores_{_score_label(args.inp, methods)}.md"
    out.write_text(_render_scores_md(results, ks, args.inp, depth), encoding="utf-8")
    print(f"Results written to {out}")


def _score_label(paths: list[Path], methods: list[str]) -> str:
    """Derive a Markdown filename from the inputs, e.g. 'I0000_top100'.

    When only one method is scored, include it ('I0000_max_top100') so a
    single-method run never overwrites the combined both-methods report.
    """
    import re

    people = sorted({m.group() for p in paths for m in re.finditer(r"I\d{4}", p.name)})
    ks = sorted({m.group(1) for p in paths for m in re.finditer(r"top(\d+)", p.name)})
    parts = ["-".join(people) or "review"]
    if len(methods) == 1:
        parts.append(methods[0])
    if len(ks) == 1:
        parts.append(f"top{ks[0]}")
    return "_".join(parts)


def _render_scores_md(results: dict, ks: list[int], paths: list[Path], depth: int) -> str:
    out = ["# Recommender — manual evaluation", ""]
    out.append(f"Whole-collection retrieval, reviewed depth {depth}. Source files:")
    out += [f"- `{p.name}`" for p in paths]

    def table(title: str, metric: int) -> list[str]:
        lines = ["", f"## {title}", "", "| Method | " + " | ".join(f"@{k}" for k in ks) + " |"]
        lines.append("| --- | " + " | ".join("---:" for _ in ks) + " |")
        for method, res in results.items():
            cells = "—" if res is None else " | ".join(f"{res[k][metric]:.2f}" for k in ks)
            lines.append(f"| {method} | {cells} |")
        return lines

    out += table("Precision@k", 0)
    out += table("nDCG@k", 1)
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    p_eval = sub.add_parser("evaluate", help="automatic Precision@k/Recall@k on the labelled pool")
    p_eval.add_argument("--person-id", default=DEFAULT_PERSON)
    p_eval.add_argument("--min-faces", type=int, default=5, help="min labelled faces to enter the macro-average")
    p_eval.set_defaults(func=cmd_evaluate)

    p_ret = sub.add_parser("retrieve", help="full-collection top-k export for manual review")
    p_ret.add_argument("--person-id", default=DEFAULT_PERSON)
    p_ret.add_argument("--top-k", type=int, default=20)
    p_ret.add_argument("--out-dir", type=Path, default=None, help="output folder (default: pipeline/recommend/reviews/)")
    p_ret.add_argument("--force", action="store_true", help="overwrite the CSVs if they already exist")
    p_ret.add_argument("--montage", action="store_true", help="also write a rank-labelled contact sheet per method for review")
    p_ret.add_argument("--montage-only", action="store_true", help="only (re)write the contact sheets; do not touch the CSVs")
    p_ret.add_argument("--ranks", default=None, help="limit the montage to a rank range, e.g. 66-85 (CSV stays full)")
    p_ret.set_defaults(func=cmd_retrieve)

    p_score = sub.add_parser("score-manual", help="compute manual Precision@k/nDCG@k from reviewed CSVs")
    p_score.add_argument("--in", dest="inp", type=Path, nargs="+", required=True, help="one or more reviewed CSVs (e.g. the per-method files)")
    p_score.add_argument("--out", type=Path, default=None, help="Markdown results path (default: scores_<person>_top<k>.md next to the CSVs)")
    p_score.set_defaults(func=cmd_score_manual)

    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    args.func(args)


if __name__ == "__main__":
    main()
