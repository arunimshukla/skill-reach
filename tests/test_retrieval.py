# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test dense semantic and hybrid retrieval scorers and RRF fusion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import reach.retrieval
from reach.models import Skill
from reach.retrieval import (
    DenseScorer,
    HybridScorer,
    build_scorer,
    compute_rrf,
    cosine_similarity,
    directional_projection,
)


def _make_skill(name: str, desc: str) -> Skill:
    """Create a minimal Skill object for testing."""
    return Skill(
        name=name,
        description=desc,
        path=Path(f"/skills/{name}"),
    )


def test_cosine_similarity_identical_vectors() -> None:
    """Verify cosine similarity of identical vectors is 1.0."""
    v = [1.0, 2.0, 3.0]
    assert pytest.approx(cosine_similarity(v, v)) == 1.0


def test_cosine_similarity_orthogonal_vectors() -> None:
    """Verify cosine similarity of orthogonal vectors is 0.0."""
    v1 = [1.0, 0.0]
    v2 = [0.0, 1.0]
    assert pytest.approx(cosine_similarity(v1, v2)) == 0.0


def test_cosine_similarity_zero_vector_returns_zero() -> None:
    """Verify cosine similarity handles zero magnitude vectors safely."""
    v1 = [0.0, 0.0]
    v2 = [1.0, 2.0]
    assert cosine_similarity(v1, v2) == 0.0


def test_directional_projection_asymmetry() -> None:
    """Verify directional projection measures proportion of target covered by candidate."""
    # Target is specialized: [1.0, 0.0]
    # Candidate is broad: [1.0, 1.0]
    target = [1.0, 0.0]
    candidate = [1.0, 1.0]

    # Target projecting onto candidate
    proj_t_to_c = directional_projection(target, candidate)
    # Candidate projecting onto target
    proj_c_to_t = directional_projection(candidate, target)

    assert proj_t_to_c != proj_c_to_t
    assert pytest.approx(proj_t_to_c) == 1.0  # Target vector fully covered by candidate
    assert pytest.approx(proj_c_to_t) == 0.5  # Candidate vector partially covered by target


def test_compute_rrf_fuses_rankings() -> None:
    """Verify Reciprocal Rank Fusion combines two independent ranking lists."""
    # List 1: A (rank 1), B (rank 2), C (rank 3)
    # List 2: B (rank 1), C (rank 2), A (rank 3)
    rankings = [
        ["skill-a", "skill-b", "skill-c"],
        ["skill-b", "skill-c", "skill-a"],
    ]
    fused = compute_rrf(rankings, k=60)

    # Expected scores:
    # skill-b: 1/(60+2) + 1/(60+1) = 1/62 + 1/61 ≈ 0.016129 + 0.016393 = 0.032522
    # skill-a: 1/(60+1) + 1/(60+3) = 1/61 + 1/63 ≈ 0.016393 + 0.015873 = 0.032266
    # skill-c: 1/(60+3) + 1/(60+2) = 1/63 + 1/62 ≈ 0.015873 + 0.016129 = 0.032002
    assert fused[0][0] == "skill-b"
    assert fused[1][0] == "skill-a"
    assert fused[2][0] == "skill-c"


def test_compute_rrf_tie_breaking_alphabetical() -> None:
    """Verify RRF breaks ties deterministically using alphabetical skill name."""
    rankings = [
        ["skill-z", "skill-a"],
        ["skill-a", "skill-z"],
    ]
    fused = compute_rrf(rankings, k=60)
    # Both have identical RRF score (1/61 + 1/62), so skill-a should precede skill-z
    assert fused[0][0] == "skill-a"
    assert fused[1][0] == "skill-z"


def test_dense_scorer_with_precomputed_vectors() -> None:
    """Verify DenseScorer ranks candidates using vector similarity."""
    target = _make_skill("pdf-parser", "Extract tables from PDF documents.")
    cand1 = _make_skill("doc-extractor", "Parse structured tables from PDF files.")
    cand2 = _make_skill("image-editor", "Crop and rotate JPEG photos.")

    vectors = {
        "pdf-parser": [1.0, 0.9, 0.0],
        "doc-extractor": [0.95, 0.85, 0.0],
        "image-editor": [0.0, 0.1, 1.0],
    }

    scorer = DenseScorer(vectors=vectors)
    ranked = scorer.rank(target, [cand1, cand2])

    assert len(ranked) == 2
    assert ranked[0][0] == "doc-extractor"
    assert ranked[1][0] == "image-editor"
    assert ranked[0][1] > ranked[1][1]


def test_dense_scorer_pairwise_similarity() -> None:
    """Verify DenseScorer computes pairwise cosine similarity across skills."""
    skills = [
        _make_skill("skill-a", "Skill A description"),
        _make_skill("skill-b", "Skill B description"),
        _make_skill("skill-c", "Skill C description"),
    ]
    vectors = {
        "skill-a": [1.0, 0.0],
        "skill-b": [0.99, 0.05],  # High cosine similarity to skill-a
        "skill-c": [0.0, 1.0],  # Orthogonal vector to skill-a
    }
    scorer = DenseScorer(vectors=vectors)
    pairs = scorer.pairwise_similarity(skills)

    # Pairs are sorted by similarity descending
    assert len(pairs) == 3  # (a,b), (a,c), (b,c)
    top_pair = pairs[0]
    assert {top_pair[0], top_pair[1]} == {"skill-a", "skill-b"}
    assert top_pair[2] > 0.95


def test_hybrid_scorer_fuses_lexical_and_dense() -> None:
    """Verify HybridScorer captures both lexical keyword matches and semantic matches."""
    target = _make_skill("pdf-tables", "Extract tabular data from PDF files.")
    # Lexical match (shares keywords "extract", "pdf")
    lexical_rival = _make_skill("pdf-tool", "Extract text strings from PDF documents.")
    # Semantic match (different words: "parse", "spreadsheets", but conceptually identical)
    semantic_rival = _make_skill("sheet-parser", "Convert document tables into spreadsheets.")
    # Unrelated
    unrelated = _make_skill("audio-player", "Play mp3 audio streams.")

    skills = [target, lexical_rival, semantic_rival, unrelated]

    # Provide vectors where sheet-parser is semantically close to pdf-tables
    vectors = {
        "pdf-tables": [1.0, 0.8, 0.0],
        "sheet-parser": [0.95, 0.80, 0.0],
        "pdf-tool": [0.5, 0.2, 0.0],
        "audio-player": [0.0, 0.0, 1.0],
    }

    hybrid = HybridScorer.from_skills_and_vectors(
        skills=skills,
        vectors=vectors,
        rrf_k=60,
    )
    candidates = [lexical_rival, semantic_rival, unrelated]
    ranked = hybrid.rank(target, candidates)

    assert len(ranked) == 3
    # Both lexical and semantic rivals should rank above unrelated
    ranked_names = [name for name, _ in ranked]
    assert ranked_names[2] == "audio-player"
    assert set(ranked_names[:2]) == {"pdf-tool", "sheet-parser"}


def test_build_scorer_factory() -> None:
    """Verify build_scorer constructs requested scorer variants."""
    skills = [
        _make_skill("tool-1", "Tool 1 description"),
        _make_skill("tool-2", "Tool 2 description"),
    ]
    bm25 = build_scorer("bm25", skills)
    assert bm25.__class__.__name__ == "Bm25Scorer"

    with pytest.raises(ValueError, match="unknown scorer"):
        build_scorer("non-existent-scorer", skills)


def test_build_neighborhood_catalogs_with_hybrid_scorer() -> None:
    """Verify build_neighborhood_catalogs respects hybrid scorer rankings."""
    from reach.catalog import build_neighborhood_catalogs

    target = _make_skill("pdf-tables", "Extract tabular data from PDF files.")
    lexical_rival = _make_skill("pdf-tool", "Extract text strings from PDF documents.")
    semantic_rival = _make_skill("sheet-parser", "Convert document tables into spreadsheets.")
    filler = _make_skill("audio-player", "Play mp3 audio streams.")

    skills = [target, lexical_rival, semantic_rival, filler]
    vectors = {
        "pdf-tables": [1.0, 0.8, 0.0],
        "sheet-parser": [0.95, 0.80, 0.0],
        "pdf-tool": [0.5, 0.2, 0.0],
        "audio-player": [0.0, 0.0, 1.0],
    }
    hybrid = HybridScorer.from_skills_and_vectors(skills, vectors, rrf_k=60)
    catalogs = build_neighborhood_catalogs(skills, size=3, rivals=2, scorer=hybrid)
    target_catalog = next(c for c in catalogs if c.id == "neighborhood:pdf-tables")
    # Verify target and both rival categories (lexical and semantic) are included
    assert set(target_catalog.skills) == {"pdf-tables", "pdf-tool", "sheet-parser"}


def test_classify_overlap_quadrant() -> None:
    """Verify dual-axis overlap quadrant classification."""
    from reach.retrieval import classify_overlap_quadrant

    assert classify_overlap_quadrant(0.8, 0.9) == "Near-Duplicate"
    assert classify_overlap_quadrant(0.7, 0.4) == "Boilerplate / Style"
    assert classify_overlap_quadrant(0.2, 0.85) == "Latent Collision"
    assert classify_overlap_quadrant(0.1, 0.3) == "Distinct"


def test_build_scorer_dense_missing_model2vec_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify dense build_scorer raises RuntimeError with pip tip when model2vec is absent."""
    skills = [_make_skill("s1", "desc1")]

    def mock_load(model_name: str) -> Any:
        msg = (
            "model2vec is required for dense semantic scoring. "
            "Install it with: pip install 'skill-reach[semantic]'"
        )
        raise RuntimeError(msg)

    monkeypatch.setattr(reach.retrieval, "_load_model2vec_model", mock_load)

    with pytest.raises(RuntimeError, match=r"pip install 'skill-reach\[semantic\]'"):
        build_scorer("dense", skills)


def test_build_scorer_hybrid_missing_model2vec_falls_back_to_bm25(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify build_scorer('hybrid') falls back to Bm25Scorer when model2vec is missing."""
    skills = [_make_skill("s1", "desc1")]

    def mock_from_skills(*args: Any, **kwargs: Any) -> Any:
        msg = "model2vec is required"
        raise RuntimeError(msg)

    monkeypatch.setattr(reach.retrieval.HybridScorer, "from_skills", mock_from_skills)

    scorer = build_scorer("hybrid", skills)
    assert scorer.__class__.__name__ == "Bm25Scorer"


def test_unit_vector_normalization() -> None:
    """Verify _unit_vector normalizes vectors to length 1.0 and handles zero vectors."""
    from reach.retrieval import _unit_vector

    norm = _unit_vector([3.0, 4.0])
    assert pytest.approx(norm) == [0.6, 0.8]

    zero = _unit_vector([0.0, 0.0, 0.0])
    assert zero == [0.0, 0.0, 0.0]


def test_dense_scorer_zero_vector_handling() -> None:
    """Verify DenseScorer handles zero-length vectors without dividing by zero."""
    s1 = _make_skill("s1", "desc1")
    s2 = _make_skill("s2", "desc2")
    scorer = DenseScorer(vectors={"s1": [0.0, 0.0], "s2": [1.0, 1.0]})

    pairs = scorer.pairwise_similarity([s1, s2])
    assert len(pairs) == 1
    assert pairs[0][2] == 0.0

    ranked = scorer.rank(s1, [s2])
    assert ranked == [("s2", 0.0)]


def test_bm25_scorer_declared_in_retrieval() -> None:
    """Verify Bm25Scorer, Scorer, K1, B, and tokenize are defined in reach.retrieval."""
    from reach.retrieval import K1, B, Bm25Scorer, Scorer, skill_text, tokenize

    assert K1 > 0
    assert 0 <= B <= 1
    s1 = _make_skill("skill-a", "python coding assistant")
    s2 = _make_skill("skill-b", "python code refactoring")
    tokens = tokenize(skill_text(s1))
    assert "python" in tokens
    assert "assistant" in tokens

    scorer = Bm25Scorer.from_skills([s1, s2])
    assert isinstance(scorer, Scorer)
    ranked = scorer.rank(s1, [s2])
    assert len(ranked) == 1
    assert ranked[0][0] == "skill-b"
    assert ranked[0][1] > 0
