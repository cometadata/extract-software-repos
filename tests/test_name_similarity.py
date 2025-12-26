"""Tests for name similarity heuristic."""

import pytest
from extract_software_repos.heuristics.name_similarity import (
    normalize_text,
    compute_containment_score,
    compute_token_overlap_score,
    compute_fuzzy_score,
    compute_name_similarity,
    NameSimilarityResult,
)


class TestNormalizeText:
    """Tests for text normalization."""

    def test_lowercase(self):
        assert "hello" in normalize_text("HELLO")

    def test_hyphens_to_spaces(self):
        tokens = normalize_text("pytorch-transformer")
        assert "pytorch" in tokens
        assert "transformer" in tokens

    def test_underscores_to_spaces(self):
        tokens = normalize_text("my_cool_repo")
        assert "my" in tokens
        assert "cool" in tokens
        assert "repo" in tokens

    def test_removes_stopwords(self):
        tokens = normalize_text("A Repository for the Implementation of Models")
        assert "a" not in tokens
        assert "the" not in tokens
        assert "for" not in tokens
        assert "of" not in tokens
        assert "repository" in tokens
        assert "implementation" in tokens
        assert "models" in tokens

    def test_removes_punctuation(self):
        tokens = normalize_text("Hello, World! (Test)")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens
        assert "," not in "".join(tokens)


class TestContainmentScore:
    """Tests for containment scoring."""

    def test_full_containment(self):
        score = compute_containment_score("transformer", "A Transformer Model")
        assert score == 1.0

    def test_partial_containment(self):
        # "pytorch transformer" partially contained in "transformer model"
        score = compute_containment_score("pytorch-transformer", "Transformer Model")
        assert 0 < score < 1

    def test_no_containment(self):
        score = compute_containment_score("xyz-abc", "Totally Different Name")
        assert score == 0.0

    def test_reverse_containment(self):
        # Short title contained in long repo name
        score = compute_containment_score("extended-isolation-forest-implementation", "Isolation Forest")
        assert score > 0


class TestTokenOverlapScore:
    """Tests for token overlap scoring."""

    def test_identical(self):
        score = compute_token_overlap_score("transformer model", "Transformer Model")
        assert score == 1.0

    def test_partial_overlap(self):
        score = compute_token_overlap_score("pytorch-transformer", "Transformer Language Model")
        # Only "transformer" overlaps
        assert 0 < score < 1

    def test_no_overlap(self):
        score = compute_token_overlap_score("xyz-abc", "Totally Different")
        assert score == 0.0

    def test_word_order_irrelevant(self):
        score1 = compute_token_overlap_score("model transformer", "Transformer Model")
        score2 = compute_token_overlap_score("transformer model", "Transformer Model")
        assert score1 == score2


class TestFuzzyScore:
    """Tests for fuzzy string matching."""

    def test_identical(self):
        score = compute_fuzzy_score("transformer", "transformer")
        assert score == 1.0

    def test_typo(self):
        score = compute_fuzzy_score("transformr", "transformer")
        assert score > 0.8

    def test_completely_different(self):
        score = compute_fuzzy_score("xyz", "abc")
        assert score < 0.5


class TestComputeNameSimilarity:
    """Tests for the main similarity function."""

    def test_exact_match(self):
        result = compute_name_similarity(
            repo_name="transformer-model",
            paper_title="Transformer Model",
        )
        assert result.matched is True
        assert result.score > 0.8

    def test_good_match(self):
        result = compute_name_similarity(
            repo_name="pytorch-transformer-lm",
            paper_title="A PyTorch Implementation of Transformer Language Models",
        )
        assert result.matched is True
        assert result.score >= 0.45

    def test_no_match(self):
        result = compute_name_similarity(
            repo_name="awesome-utils",
            paper_title="Deep Learning for Natural Language Processing",
        )
        assert result.matched is False
        assert result.score < 0.45

    def test_custom_threshold(self):
        result = compute_name_similarity(
            repo_name="transformer",
            paper_title="Transformer Model",
            threshold=0.9,
        )
        # Might not meet high threshold
        assert result.score < 1.0

    def test_missing_repo_name(self):
        result = compute_name_similarity(
            repo_name=None,
            paper_title="Some Paper",
        )
        assert result.matched is False
        assert result.skipped is True
        assert result.skip_reason == "no_repo_name"

    def test_missing_title(self):
        result = compute_name_similarity(
            repo_name="some-repo",
            paper_title=None,
        )
        assert result.matched is False
        assert result.skipped is True
        assert result.skip_reason == "no_paper_title"

    def test_eif_example(self):
        # Real example from the dataset
        result = compute_name_similarity(
            repo_name="eif",
            paper_title="Extended Isolation Forest",
        )
        # "eif" is an acronym, might have low score but check it runs
        assert isinstance(result.score, float)
