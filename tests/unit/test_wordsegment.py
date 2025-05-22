# -*- coding: utf-8 -*-
"""
Tests for the wordsegment package.

These tests cover basic functionality, edge cases, and security aspects.
"""

# Standard library imports
import math
from unittest.mock import patch

# Third-party imports
import pytest

# Local application imports
from wordsegment.constants import SecurityConstants
from wordsegment.wordsegment import Segmenter


# Fixture to initialize a loaded segmenter for tests
@pytest.fixture(scope="module")
def loaded_segmenter():
    """Create and return a loaded Segmenter instance."""
    segmenter = Segmenter()
    segmenter.load()
    return segmenter


# Basic functionality tests
class TestBasicFunctionality:
    """Tests for basic functionality of the wordsegment package."""

    def test_unigrams(self, loaded_segmenter):
        """Test that unigrams are loaded correctly."""
        assert "test" in loaded_segmenter.unigrams
        assert isinstance(loaded_segmenter.unigrams["test"], float)

    def test_bigrams(self, loaded_segmenter):
        """Test that bigrams are loaded correctly."""
        assert "in the" in loaded_segmenter.bigrams
        assert isinstance(loaded_segmenter.bigrams["in the"], float)

    def test_words_list(self, loaded_segmenter):
        """Test that word list is loaded correctly."""
        assert len(loaded_segmenter.words) > 0
        assert isinstance(loaded_segmenter.words[0], str)
        assert isinstance(loaded_segmenter.words[-1], str)


class TestTextCleaning:
    """Tests for text cleaning functionality."""

    def test_clean_with_special_chars(self):
        """Test cleaning text with special characters."""
        segmenter = Segmenter()
        assert segmenter.clean("Can't buy me love!") == "cantbuymelove"

    def test_clean_with_mixed_case(self):
        """Test cleaning text with mixed case."""
        segmenter = Segmenter()
        assert segmenter.clean("ThIsIsA TeSt") == "thisisatest"

    def test_clean_empty_string(self):
        """Test cleaning an empty string."""
        segmenter = Segmenter()
        assert segmenter.clean("") == ""

    def test_clean_oversized_input(self):
        """Test cleaning text that exceeds the maximum length."""
        segmenter = Segmenter()
        max_length = SecurityConstants.MAX_TEXT_LENGTH.value
        oversized = "a" * (max_length + 100)
        result = segmenter.clean(oversized)
        assert len(result) == max_length


class TestSegmentation:
    """Tests for text segmentation functionality."""

    def test_segment_simple_words(self, loaded_segmenter):
        """Test segmentation of simple concatenated words."""
        text = "choosespain"
        expected = ["choose", "spain"]
        result = loaded_segmenter.segment(text)
        assert result == expected

    def test_segment_short_phrase(self, loaded_segmenter):
        """Test segmentation of a short phrase."""
        text = "thisisatest"
        expected = ["this", "is", "a", "test"]
        result = loaded_segmenter.segment(text)
        assert result == expected

    def test_segment_ambiguous_words(self, loaded_segmenter):
        """Test segmentation of ambiguous word combinations."""
        test_cases = [
            ("whorepresents", ["who", "represents"]),
            ("expertsexchange", ["experts", "exchange"]),
            ("speedofart", ["speed", "of", "art"]),
        ]
        for text, expected in test_cases:
            assert loaded_segmenter.segment(text) == expected

    def test_segment_and_title(self, loaded_segmenter):
        """Test the segment_and_title function."""
        text = "thisisatest"
        expected = ["This", "Is", "A", "Test"]
        result = loaded_segmenter.segment_capitalized(text)
        assert result == expected


class TestChunkSizeHandling:
    """Tests for chunk size handling."""

    def test_set_chunk_size_valid(self):
        """Test setting a valid chunk size."""
        segmenter = Segmenter()
        valid_size = 1000
        segmenter.set_chunk_size(valid_size)
        assert segmenter.chunk_size == valid_size

    def test_set_chunk_size_invalid(self):
        """Test setting an invalid chunk size."""
        segmenter = Segmenter()
        invalid_sizes = [-1, 0, "string", SecurityConstants.MAX_CHUNK_SIZE.value + 1]
        for size in invalid_sizes:
            with pytest.raises((ValueError, TypeError)):
                segmenter.set_chunk_size(size)

    def test_segment_with_custom_chunk_size(self, loaded_segmenter):
        """Test segmentation with a custom chunk size."""
        text = "thisisatest"
        custom_size = 5
        result = loaded_segmenter.segment(text, chunk_size=custom_size)
        assert result == ["this", "is", "a", "test"]


class TestErrorHandling:
    """Tests for error handling."""

    def test_validate_path_nonexistent(self):
        """Test validation of a nonexistent path."""
        segmenter = Segmenter()
        nonexistent_path = "/path/to/nonexistent/file.txt"
        with pytest.raises(FileNotFoundError):
            segmenter._validate_path(nonexistent_path)

    @patch("os.path.exists")
    @patch("os.path.islink")
    @patch("os.path.isfile")
    @patch("os.path.getsize")
    def test_validate_path_symlink(
        self, mock_getsize, mock_isfile, mock_islink, mock_exists
    ):
        """Test validation of a symlink path."""
        segmenter = Segmenter()
        symlink_path = "/path/to/symlink.txt"

        # Configure mocks
        mock_exists.return_value = True
        mock_islink.return_value = True
        mock_isfile.return_value = True
        mock_getsize.return_value = 100

        with pytest.raises(ValueError) as excinfo:
            segmenter._validate_path(symlink_path)
        assert "Symlinks not allowed" in str(excinfo.value)

    @patch("os.path.exists")
    @patch("os.path.islink")
    @patch("os.path.isfile")
    @patch("os.path.getsize")
    def test_validate_path_too_large(
        self, mock_getsize, mock_isfile, mock_islink, mock_exists
    ):
        """Test validation of a file that is too large."""
        segmenter = Segmenter()
        large_file_path = "/path/to/large_file.txt"

        # Configure mocks
        mock_exists.return_value = True
        mock_islink.return_value = False
        mock_isfile.return_value = True
        mock_getsize.return_value = SecurityConstants.MAX_FILE_SIZE.value + 1

        with pytest.raises(ValueError) as excinfo:
            segmenter._validate_path(large_file_path)
        assert "File too large" in str(excinfo.value)

    def test_score_without_loading(self):
        """Test scoring without loading data first."""
        segmenter = Segmenter()
        # Don't call segmenter.load() to test the error case

        with pytest.raises(RuntimeError):
            segmenter.score("test")


class TestAdvancedFeatures:
    """Tests for advanced features."""

    def test_score_known_word(self, loaded_segmenter):
        """Test scoring a known word."""
        word = "the"  # Common word
        score = loaded_segmenter.score(word)
        assert score > 0
        assert math.isfinite(score)

    def test_score_unknown_word(self, loaded_segmenter):
        """Test scoring an unknown word."""
        word = "xyzabc123"  # Unlikely to be in vocabulary
        score = loaded_segmenter.score(word)
        assert score > 0  # Should apply smoothing
        assert score < 0.001  # Should be a very small probability
        assert math.isfinite(score)

    def test_bigram_probability(self, loaded_segmenter):
        """Test bigram probability calculation."""
        first = "in"
        second = "the"

        # Score of "the" given "in"
        score_conditional = loaded_segmenter.score(second, first)
        # Score of "the" without context
        score_unconditional = loaded_segmenter.score(second)

        # Conditional probability should be higher
        assert score_conditional > score_unconditional


def test_malformed_input():
    """Test handling of malformed input."""
    segmenter = Segmenter()
    segmenter.load()
    inputs = [None, 123, [], {}]
    for inp in inputs:
        try:
            result = segmenter.segment(str(inp))
            # If it doesn't raise an exception, it should return a list
            assert isinstance(result, list)
        except Exception as e:
            # If an exception is raised, it should be a ValueError or TypeError
            assert isinstance(e, (ValueError, TypeError))


def test_dos_attack_resistance():
    """Test resistance to potential denial of service attacks."""
    segmenter = Segmenter()
    segmenter.load()
    # Create input designed to be challenging for segmentation algorithms
    pathological_input = "a" * 100 + "b" * 100 + "c" * 100
    result = segmenter.segment(pathological_input)
    # Just verify it completes without error
    assert isinstance(result, list)


def test_unicode_handling():
    """Test handling of Unicode characters."""
    segmenter = Segmenter()
    segmenter.load()
    # Text with various Unicode characters
    unicode_text = "café蘋果привет"
    result = segmenter.segment(unicode_text)
    assert isinstance(result, list)
