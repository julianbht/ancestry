"""Tests for the face_recognition hpopt: the leave-one-out metric, the Fbeta
objective weighting, and that the Optuna search actually runs end-to-end.

The embedding models (DeepFace / InsightFace) and W&B are stubbed out, so this
downloads nothing and runs anywhere — it confirms the search *wiring* and the
*scoring logic*, not the models themselves. The model seam is the build callables
in search.py (DeepFaceMethod / InsightFaceMethod); we replace them with a fake
that returns precomputed embeddings.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from pipeline.experiments.face_recognition import search
from pipeline.experiments.face_recognition.metrics import evaluate_loo
from pipeline.experiments.face_recognition.search import _fbeta, make_objective
from pipeline.experiments.study import create_study
from pipeline.face_recognition.gallery import MatchedFace


def _match(person_id: str | None, name: str) -> MatchedFace:
    return MatchedFace(person_id=person_id, age=None, crop_path=Path(name), gt_key=f"{name}.jpg")


# --- evaluate_loo: the leave-one-out scoring logic (pure numpy, no models) ---


def test_evaluate_loo_perfect_separation_all_known():
    """Two people, two well-separated crops each: every held-out query finds its
    own person nearest → recall and precision both 1.0, nothing mislabeled."""
    matches = [_match("alice", "a0"), _match("alice", "a1"), _match("bob", "b0"), _match("bob", "b1")]
    embeddings = {
        "a0": np.array([1.0, 0.0]), "a1": np.array([1.0, 0.01]),
        "b0": np.array([0.0, 1.0]), "b1": np.array([0.01, 1.0]),
    }

    result = evaluate_loo(matches, embeddings, "cosine", threshold=0.5)

    assert result.recall == 1.0
    assert result.precision == 1.0
    assert result.n_known_queries == 4
    assert result.n_negative_queries == 0
    assert result.n_emitted_labels == 4


def test_evaluate_loo_threshold_too_tight_emits_nothing():
    """A threshold below every nearest-neighbour distance labels nothing →
    recall and precision are both 0 (this is what stops the optimiser cheating)."""
    matches = [_match("alice", "a0"), _match("alice", "a1"), _match("bob", "b0"), _match("bob", "b1")]
    embeddings = {
        "a0": np.array([1.0, 0.0]), "a1": np.array([0.7, 0.7]),
        "b0": np.array([0.0, 1.0]), "b1": np.array([-0.7, 0.7]),
    }

    result = evaluate_loo(matches, embeddings, "cosine", threshold=0.05)

    assert result.n_emitted_labels == 0
    assert result.recall == 0.0
    assert result.precision == 0.0


def test_evaluate_loo_negatives_charge_precision_and_specificity():
    """Single-photo and unidentified faces are 'negative' queries: any label they
    get is a false positive. With a loose threshold both get labeled, so precision
    drops while recall on the genuine known pair stays 1.0."""
    matches = [
        _match("alice", "a0"), _match("alice", "a1"),  # known pair
        _match("carol", "c0"),                         # single photo → negative
        _match(None, "u0"),                            # unidentified → negative
    ]
    embeddings = {
        "a0": np.array([1.0, 0.0, 0.0]), "a1": np.array([1.0, 0.0, 0.0]),
        "c0": np.array([0.0, 1.0, 0.0]),
        "u0": np.array([0.0, 0.0, 1.0]),
    }

    result = evaluate_loo(matches, embeddings, "cosine", threshold=1.5)

    assert result.n_known_queries == 2
    assert result.n_negative_queries == 2
    assert result.recall == 1.0           # both alice queries matched correctly
    assert result.n_emitted_labels == 4   # 2 correct + 2 false positives
    assert result.precision == 0.5        # 2 correct / 4 emitted
    assert result.specificity == 0.0      # neither negative correctly rejected


# --- _fbeta: the precision/recall trade-off used as the objective ---


def test_fbeta_zero_when_either_component_zero():
    assert _fbeta(0.0, 0.5, 0.5) == 0.0
    assert _fbeta(0.5, 0.0, 0.5) == 0.0


def test_fbeta_perfect_is_one():
    assert _fbeta(1.0, 1.0, 0.5) == pytest.approx(1.0)


def test_fbeta_below_one_favours_precision():
    """beta < 1 should score a high-precision/low-recall config above the mirror
    image (the whole point of weighting precision ~2x for gallery purity)."""
    assert _fbeta(0.9, 0.3, 0.5) > _fbeta(0.3, 0.9, 0.5)


# --- the Optuna objective, end-to-end with the model + W&B stubbed out ---


def _make_dataset(rng: np.random.Generator) -> tuple[list[MatchedFace], dict[str, np.ndarray]]:
    """A small, well-separated synthetic gallery: 3 identified people (alice/bob
    with 3 crops, carol with 2) plus 2 unidentified faces. Each person sits on its
    own random direction with tiny within-person noise, so nearest-neighbour LOO is
    meaningful and within-person distance falls below every searched threshold."""
    dim = 64
    specs = {"alice": 3, "bob": 3, "carol": 2}
    bases = {pid: rng.standard_normal(dim) for pid in specs}
    matches: list[MatchedFace] = []
    embeddings: dict[str, np.ndarray] = {}
    for pid, n in specs.items():
        for i in range(n):
            name = f"{pid}_{i}"
            embeddings[name] = (bases[pid] + 0.01 * rng.standard_normal(dim)).astype(np.float32)
            matches.append(_match(pid, name))
    for i in range(2):
        name = f"unknown_{i}"
        embeddings[name] = rng.standard_normal(dim).astype(np.float32)
        matches.append(_match(None, name))
    return matches, embeddings


def _fake_method_class(embeddings: dict[str, np.ndarray], built: list[object]) -> type:
    """A stand-in for DeepFaceMethod / InsightFaceMethod: constructs with any args
    (no model load) and returns the precomputed embedding for each crop. Appends to
    `built` on every construction so the embedding cache can be asserted."""

    class _FakeMethod:
        def __init__(self, *args, **kwargs):
            built.append(object())
            self.name = "fake"

        def embed(self, crop_path: Path) -> np.ndarray | None:
            return embeddings.get(str(crop_path))

    return _FakeMethod


@pytest.fixture
def _stub_models_and_wandb(monkeypatch):
    """Replace both method builders with one fake and silence W&B. Returns the
    list that records every fake-method construction."""
    rng = np.random.default_rng(0)
    matches, embeddings = _make_dataset(rng)
    built: list[object] = []
    fake = _fake_method_class(embeddings, built)
    monkeypatch.setattr(search, "DeepFaceMethod", fake)
    monkeypatch.setattr(search, "InsightFaceMethod", fake)
    monkeypatch.setattr(search, "wandb", MagicMock())
    return matches, built


@pytest.mark.parametrize("method", ["deepface", "insightface"])
def test_objective_runs_for_both_methods(method, _stub_models_and_wandb):
    matches, _built = _stub_models_and_wandb

    study = create_study(f"test-{method}", "maximize")
    study.optimize(make_objective(matches, method, "proj", "group"), n_trials=6)

    assert len(study.trials) == 6
    # Well-separated data → at least one config scores above zero.
    assert study.best_value > 0.0
    best = study.best_trial
    for attr in (
        "precision", "recall", "specificity",
        "n_known_queries", "n_negative_queries", "n_emitted_labels",
    ):
        assert attr in best.user_attrs


def test_embedding_cache_builds_once_per_combo(_stub_models_and_wandb):
    """Two trials sharing the same embedding-defining params (model + backend) but
    differing only in threshold must embed once — the cache is what keeps the slow
    part from re-running on every trial."""
    matches, built = _stub_models_and_wandb

    study = create_study("test-cache", "maximize")
    objective = make_objective(matches, "deepface", "proj", "group")
    common = {"model_name": "ArcFace", "detector_backend": "skip", "distance_metric": "cosine"}
    study.enqueue_trial({**common, "threshold_cosine": 0.3})
    study.enqueue_trial({**common, "threshold_cosine": 0.7})
    study.optimize(objective, n_trials=2)

    assert len(built) == 1
