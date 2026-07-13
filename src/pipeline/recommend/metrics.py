"""Precision@k / Recall@k for the face recommender.

Two evaluation methods, deliberately different in where the relevance labels
come from (the project checklist asks for two):

  A. Automatic (this module) — uses the Label Studio ground truth. We hold part
     of a person's labelled faces out as the query and try to retrieve the rest
     from the labelled pool, so every retrieved face has a known person_id and
     precision/recall are computed without a human. Because using all of a
     person's faces as the query would leave none to find, we average over many
     random query/positive splits (Monte-Carlo cross-validation).

  B. Manual (see __main__'s `retrieve` mode) — runs retrieval over the *whole*
     collection, most of which is unlabelled, and a human marks the top-k. That
     measures real-world precision on faces the ground truth never covered.

Definitions, per query:
    Precision@k = (relevant in top-k) / k
    Recall@k    = (relevant in top-k) / (total relevant in the pool)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pipeline.recommend.index import FaceIndex
from pipeline.recommend.retrieval import Strategy, recommend


def precision_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    top = ranked[:k]
    return sum(i in relevant for i in top) / k if top else 0.0


def recall_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    top = ranked[:k]
    return sum(i in relevant for i in top) / len(relevant)


def ndcg_at_k(hits: list[bool], k: int) -> float:
    """nDCG@k for binary relevance, normalised by the hits present in the top-k.

    DCG rewards correct items near the top (1/log2(rank+1)); the ideal DCG packs
    all the top-k hits into the first slots. nDCG=1 means every hit found was
    ranked above every miss. Self-normalising by the hits in the top-k, so it
    needs no knowledge of the (unknown) total relevant count — apt for the manual
    whole-collection evaluation.
    """
    import math

    top = hits[:k]
    dcg = sum(1.0 / math.log2(i + 2) for i, hit in enumerate(top) if hit)
    n_rel = sum(top)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_rel))
    return dcg / idcg if idcg > 0 else 0.0


@dataclass
class EvalResult:
    person_id: str
    strategy: Strategy
    n_queries: int  # number of split repetitions averaged
    precision: dict[int, float]
    recall: dict[int, float]


def evaluate_person(
    index: FaceIndex,
    person_id: str,
    strategy: Strategy,
    ks: list[int],
    n_splits: int = 50,
    query_frac: float = 0.5,
    seed: int = 0,
) -> EvalResult:
    """Monte-Carlo Precision@k/Recall@k for one person on the labelled pool."""
    labeled_pool = index.labeled_indices()
    person_faces = index.indices_for_person(person_id)
    if len(person_faces) < 2:
        raise ValueError(f"{person_id} has <2 labelled faces; cannot evaluate retrieval")

    rng = np.random.default_rng(seed)
    n_query = max(1, int(round(len(person_faces) * query_frac)))
    n_query = min(n_query, len(person_faces) - 1)  # always leave ≥1 positive

    prec = {k: [] for k in ks}
    rec = {k: [] for k in ks}
    for _ in range(n_splits):
        shuffled = rng.permutation(person_faces)
        query = shuffled[:n_query].tolist()
        positives = set(shuffled[n_query:].tolist())

        # Candidate pool = labelled faces minus the query faces.
        ranked = [i for i, _ in recommend(index, query, strategy, top_k=max(ks), pool=labeled_pool)]
        for k in ks:
            prec[k].append(precision_at_k(ranked, positives, k))
            rec[k].append(recall_at_k(ranked, positives, k))

    return EvalResult(
        person_id=person_id,
        strategy=strategy,
        n_queries=n_splits,
        precision={k: float(np.mean(prec[k])) for k in ks},
        recall={k: float(np.mean(rec[k])) for k in ks},
    )
