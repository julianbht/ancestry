"""Leave-one-out evaluation of the face-recognition gallery against the
face_annotation ground truth.

The ground truth is small (a few dozen identified people, most with only a
handful of photos), so a fixed train/test split would waste most of it. LOO
instead reuses every labeled face as both gallery and query exactly once:

  - A face belonging to a person with >= 2 matched GT faces is a "known"
    query: hold it out, search every *other* matched face (regardless of
    person), and check whether the nearest one within `threshold` belongs to
    the same person. A correct hit is a true positive (feeds recall); a hit on
    the wrong person, or a miss, is not.
  - A face belonging to a person with only 1 matched GT face, or a GT face
    that was never assigned a person_id at all, is a "negative" query: there
    is no other reference for that identity, so the correct answer is
    "unknown" (nearest match beyond `threshold`). Any label emitted here is a
    false positive.

The objective is the micro-averaged precision/recall trade-off (Fbeta, computed
in search.py). Precision = correct labels / all emitted labels, so it is charged
for *both* false-positive sources: a negative query that gets labelled, and a
known query matched to the wrong person. Recall = correct labels / all known
queries, which stops the optimiser from cheating by labelling almost nothing
(a tight threshold gives high precision but near-zero recall, hence low Fbeta).
specificity (correct rejections / all negative queries) is still reported as a
diagnostic, but is no longer part of the objective: precision now carries the
"don't mislabel" signal, in a form that also catches wrong-person hits.

This intentionally cannot validate single-photo people's *own* identity (no
second photo exists to hold out against), even though build_gallery() does
use their one photo as a real 1-shot reference in production — see
pipeline/face_recognition/gallery.py.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from pipeline.face_recognition.gallery import MatchedFace
from pipeline.face_recognition.recognizer import distance


@dataclass
class LOOResult:
    recall: float  # correct hits / known queries          (TP / (TP + FN))
    precision: float  # correct hits / all emitted labels   (TP / (TP + FP))
    specificity: float  # correct rejections / negative queries — diagnostic only
    n_known_queries: int
    n_negative_queries: int
    n_emitted_labels: int  # queries assigned an identity (best match within threshold)


def evaluate_loo(
    matches: list[MatchedFace],
    embeddings: dict[str, np.ndarray | None],
    distance_metric: str,
    threshold: float,
) -> LOOResult:
    """Run one leave-one-out pass over every match for a single
    (distance_metric, threshold) configuration.

    Faces that failed to embed (embeddings[...] is None) are NOT dropped: every
    GT-matched crop is a real face (the GT says so), so a failed embedding is a
    genuine detection failure the search should be charged for, not hidden. A
    failed query emits no label (best_dist stays inf), exactly as in production,
    so a failed *known* query becomes an unrecovered miss that drags recall down
    — which is what gives pad_ratio/det_size/det_thresh pressure to reduce
    failures. The known-vs-negative partition is computed over the full match set
    (not just the embeddable subset) so a failure can't silently reclassify a
    query between trials, keeping objective values comparable."""
    # Stable partition: count GT faces per person over ALL matches, not just the
    # ones that embedded this trial.
    faces_per_person: dict[str, int] = defaultdict(int)
    for m in matches:
        if m.person_id:
            faces_per_person[m.person_id] += 1

    known_correct = 0  # true positives: known query matched to the right person
    known_total = 0
    negative_correct = 0  # negative query correctly left unknown
    negative_total = 0
    emitted = 0  # queries assigned any identity (best match within threshold)

    for i, m in enumerate(matches):
        is_known_query = m.person_id is not None and faces_per_person[m.person_id] >= 2

        query_emb = embeddings.get(str(m.crop_path))
        best_pid: str | None = None
        best_dist = float("inf")
        # A failed query (query_emb is None) can't match anything; best_dist stays
        # inf, so it emits no label — a miss for known queries. References must
        # themselves have embedded (a failed crop is no reference, as in production).
        if query_emb is not None:
            for j, other in enumerate(matches):
                if j == i or other.person_id is None:
                    continue
                other_emb = embeddings.get(str(other.crop_path))
                if other_emb is None:
                    continue
                d = distance(query_emb, other_emb, distance_metric)
                if d < best_dist:
                    best_dist = d
                    best_pid = other.person_id

        labelled = best_dist <= threshold  # we emit best_pid as this face's identity
        if labelled:
            emitted += 1

        if is_known_query:
            known_total += 1
            if labelled and best_pid == m.person_id:
                known_correct += 1
        else:
            negative_total += 1
            if not labelled:
                negative_correct += 1

    recall = known_correct / known_total if known_total else 0.0
    precision = known_correct / emitted if emitted else 0.0
    specificity = negative_correct / negative_total if negative_total else 0.0

    return LOOResult(
        recall=recall,
        precision=precision,
        specificity=specificity,
        n_known_queries=known_total,
        n_negative_queries=negative_total,
        n_emitted_labels=emitted,
    )
