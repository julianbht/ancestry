"""The two content-based retrieval strategies the recommender compares.

Both take a set of query faces (a person's reference embeddings) and rank every
other face in the collection by similarity. They differ only in how the query
set is collapsed into a score:

  - max-similarity ("nearest reference"): a candidate scores as its single best
    cosine similarity to any query face. Sensitive to one good match — strong
    when the person looks very different across photos (age, pose), since any one
    close reference suffices.

  - centroid ("prototype"): the query faces are averaged into one prototype
    vector and candidates are scored against that. Smooths over noisy individual
    references but blurs genuine intra-person variation.

Embeddings are already L2-normalised in the index, so a dot product is cosine
similarity. Query faces themselves are excluded from their own results.
"""

from __future__ import annotations

import numpy as np

from pipeline.recommend.index import FaceIndex

Strategy = str  # "max" | "centroid"
STRATEGIES: tuple[Strategy, ...] = ("max", "centroid")


def _scores(index: FaceIndex, query_indices: list[int], strategy: Strategy) -> np.ndarray:
    query = index.embeddings[query_indices]  # (m, D), already normalised
    if strategy == "max":
        return (index.embeddings @ query.T).max(axis=1)  # (N,)
    if strategy == "centroid":
        centroid = query.mean(axis=0)
        norm = np.linalg.norm(centroid)
        centroid = centroid / norm if norm else centroid
        return index.embeddings @ centroid  # (N,)
    raise ValueError(f"unknown strategy: {strategy!r}")


def recommend(
    index: FaceIndex,
    query_indices: list[int],
    strategy: Strategy,
    top_k: int,
    pool: list[int] | None = None,
) -> list[tuple[int, float]]:
    """Rank faces for a query person; return [(face_index, score)] best first.

    pool restricts which faces may be returned (e.g. the labelled subset for
    automatic evaluation). Query faces are always excluded from their own ranking.
    """
    scores = _scores(index, query_indices, strategy)

    candidates = np.array(pool if pool is not None else range(len(index)))
    query_set = set(query_indices)
    candidates = np.array([c for c in candidates if c not in query_set])

    ranked = candidates[np.argsort(scores[candidates])[::-1]][:top_k]
    return [(int(i), float(scores[i])) for i in ranked]
