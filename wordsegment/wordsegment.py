"""English Word Segmentation in Python

Word segmentation is the process of dividing a phrase without spaces back
into its constituent parts. For example, consider a phrase like "thisisatest".
For humans, it's relatively easy to parse. This module makes it easy for
machines too. Use `segment` to parse a phrase into its parts:

# >>> from wordsegment import load, segment
# >>> load()
# >>> segment('thisisatest')
['this', 'is', 'a', 'test']

In the code, 1024908267229 is the total number of words in the corpus. A
subset of this corpus is found in unigrams.txt and bigrams.txt which
should accompany this file. A copy of these files may be found at
http://norvig.com/ngrams/ under the names count_1w.txt and count_2w.txt
respectively.

Copyright (c) 2016 by Grant Jenks

Based on code from the chapter "Natural Language Corpus Data"
from the book "Beautiful Data" (Segaran and Hammerbacher, 2009)
http://oreilly.com/catalog/9780596157111/

Original Copyright (c) 2008-2009 by Peter Norvig

"""

import logging
import math
import os
from typing import Optional, Generator, Dict, List, Tuple

from wordsegment.constants import (
    SecurityConstants,
    SegmenterConstants,
    DirectoryConstants
)

# Configure logging (only basename for paths)
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('wordsegment')


class Segmenter:
    """English word segmentation based on unigrams and bigrams with stupid backoff."""

    def __init__(self):
        """Initialize the segmenter with empty data structures."""
        self.unigrams: Dict[str, float] = {}
        self.bigrams: Dict[str, float] = {}
        self.total: float = 0.0
        self.limit: int = SegmenterConstants.LIMIT.value
        self.words: List[str] = []
        self.chunk_size: int = SecurityConstants.DEFAULT_CHUNK_SIZE.value

    def _validate_path(self, path: Optional[str]) -> Optional[str]:
        """Validate that a file path is safe to use.

        Args:
            path: The path to validate

        Returns:
            The validated real path

        Raises:
            ValueError: If path is invalid or insecure
            FileNotFoundError: If path does not exist
        """
        if not path:
            return None

        # Absolute and real path resolution
        abs_path = os.path.abspath(path)
        real_path = os.path.realpath(path)

        # Check existence
        if not os.path.exists(real_path):
            raise FileNotFoundError(f"Path does not exist: {os.path.basename(real_path)}")

        # Disallow symlinks
        if os.path.islink(abs_path) or os.path.islink(real_path):
            raise ValueError(f"Symlinks not allowed: {os.path.basename(real_path)}")

        # Check size limit
        max_size = SecurityConstants.MAX_FILE_SIZE.value
        if os.path.isfile(real_path) and os.path.getsize(real_path) > max_size:
            raise ValueError(f"File too large: {os.path.basename(real_path)}")

        return real_path

    def load(
            self,
            unigrams_path: str = str(DirectoryConstants.UNIGRAMS_FILE_PATH.value),
            bigrams_path: str = str(DirectoryConstants.BIGRAMS_FILE_PATH.value),
            words_path: str = str(DirectoryConstants.WORDS_FILE_PATH.value)
    ) -> None:
        """Load unigram and bigram counts, plus word list.

        Args:
            unigrams_path: Path to the unigrams file
            bigrams_path: Path to the bigrams file
            words_path: Path to the words file

        Raises:
            FileNotFoundError: If unigram file not found or is empty
        """
        try:
            # Validate and get paths
            uni_path = self._validate_path(unigrams_path)
            bi_path = self._validate_path(bigrams_path)
            wd_path = self._validate_path(words_path)

            # Unigram counts
            if uni_path is None:
                raise FileNotFoundError("Unigrams path is None")
            self.unigrams = self._parse_counts(uni_path)
            if not self.unigrams:
                raise FileNotFoundError(
                    f"Unigram file not found or empty: {os.path.basename(uni_path)}"
                )

            # Bigram counts
            if bi_path is None:
                raise FileNotFoundError("Bigrams path is None")
            self.bigrams = self._parse_counts(bi_path)

            # Total words for probability denominator
            self.total = SegmenterConstants.TOTAL.value

            # Word list (optional)
            if wd_path is None:
                raise FileNotFoundError("Words path is None")
            self.words = self._load_words(wd_path)

            logger.info(
                f"Loaded {len(self.unigrams)} unigrams, "
                f"{len(self.bigrams)} bigrams, "
                f"{len(self.words)} words"
            )
        except (IOError, OSError) as e:
            logger.error(f"Error loading language model files: {e}")
            raise

    def _load_words(self, path: str) -> List[str]:
        """Load words from a file.

        Args:
            path: Path to the words file

        Returns:
            List of words
        """
        words: List[str] = []
        if not os.path.isfile(path):
            return words

        max_line_length = SecurityConstants.MAX_LINE_LENGTH.value
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if len(line) > max_line_length:
                        logger.debug(f"Skipping long line in words file ({len(line)} chars)")
                        continue
                    word = line.strip()
                    if word:
                        words.append(word)
        except (IOError, OSError) as e:
            logger.error(f"Error reading words file {os.path.basename(path)}: {e}")

        return words

    @staticmethod
    def _parse_counts(path: str) -> Dict[str, float]:
        """Parse a two-column file with words/phrases and their occurrence counts.

        Args:
            path: Path to the counts file with format "word/phrase <tab> count"

        Returns:
            A dictionary mapping words/phrases to their frequency counts

        Raises:
            RuntimeError: If the file contains too many entries
            IOError: If there's an error reading the file
        """
        counts: Dict[str, float] = {}
        if not os.path.isfile(path):
            logger.warning(f"Count file not found: {os.path.basename(path)}")
            return counts

        entry_count = 0
        max_entries = SecurityConstants.MAX_COUNT_ENTRIES.value
        max_line_length = SecurityConstants.MAX_LINE_LENGTH.value

        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    # Skip overly long lines for security
                    if len(line) > max_line_length:
                        logger.warning("Skipping overly long line in counts file")
                        continue

                    parts = Segmenter._parse_count_line(line)
                    if not parts:
                        continue

                    word, count_str = parts

                    # Validate that count is numeric (allowing for decimal values)
                    if not count_str.replace('.', '', 1).isdigit():
                        logger.debug(f"Invalid count value: {count_str}")
                        continue

                    counts[word] = float(count_str)
                    entry_count += 1

                    # Security check for file size
                    if entry_count > max_entries:
                        raise RuntimeError(
                            f"Count file has too many entries (limit: {max_entries})"
                        )

        except (IOError, OSError) as e:
            logger.error(f"Error reading counts file {os.path.basename(path)}: {e}")
            raise
        except ValueError as e:
            logger.error(f"Error parsing value in counts file {os.path.basename(path)}: {e}")

        return counts

    @staticmethod
    def _parse_count_line(line: str) -> Optional[Tuple[str, str]]:
        """Parse a line from a counts file.

        Args:
            line: Line to parse

        Returns:
            Tuple of (word, count) or None if parsing failed
        """
        # Try different delimiter patterns
        parts = line.strip().split('\t')
        if len(parts) == 2:
            return parts[0], parts[1]

        # Try double space delimiter
        parts = line.strip().split('  ')
        if len(parts) == 2:
            return parts[0], parts[1]

        # Last resort: find last space as delimiter
        idx = line.strip().rstrip().rfind(' ')
        if idx > 0:
            return line[:idx].strip(), line[idx:].strip()

        logger.debug(f"Could not parse line: {line.strip()}")
        return None

    def score(self, word: str, previous: Optional[str] = None) -> float:
        """Calculate the conditional probability score of a word.

        If previous word is provided and exists in unigrams, uses bigram probability.
        Otherwise uses unigram probability with Laplace smoothing for unknown words.

        Args:
            word: The word to score
            previous: Optional previous word for conditional probability

        Returns:
            Conditional probability score of the word

        Raises:
            RuntimeError: If language data has not been loaded
        """
        # Check if data is loaded
        if self.total == 0:
            raise RuntimeError("WordSegment data not loaded. Call load() first.")

        # Case 1: No previous word or previous word not in vocabulary
        if previous is None or previous not in self.unigrams:
            # If word exists in vocabulary, return its probability
            if word in self.unigrams:
                return self.unigrams.get(word, 0.0) / self.total

            # For unknown words, apply Laplace smoothing with factor of 10
            smoothing_factor = 10.0
            return smoothing_factor / (self.total * smoothing_factor * len(word))

        # Case 2: Check if bigram exists
        bigram = f"{previous} {word}"
        if bigram in self.bigrams:
            # P(current|previous) = P(previous,current) / P(previous)
            return self.bigrams[bigram] / (self.total * self.score(previous))

        # Case 3: Fall back to unigram probability
        return self.score(word)

    @classmethod
    def clean(cls, text: str) -> str:
        """Clean text by converting to lowercase and removing non-alphabet characters.

        Args:
            text: The input text to clean

        Returns:
            Cleaned text containing only lowercase alphabet characters

        Notes:
            - If input text exceeds maximum length, it will be truncated
            - Empty input returns empty string
        """
        # Handle empty input
        if not text:
            return ""

        # Truncate text if it exceeds maximum length for security
        max_length = SecurityConstants.MAX_TEXT_LENGTH.value
        if len(text) > max_length:
            logger.warning(
                f"Input text truncated from {len(text)} to {max_length} chars"
            )
            text = text[:max_length]

        # Filter text to keep only alphabet characters and convert to lowercase
        allowed_chars = SegmenterConstants.ALPHABET.value
        return ''.join(ch for ch in text.lower() if ch in allowed_chars)

    def divide(self, text: str) -> Generator[Tuple[str, str], None, None]:
        """Generate all possible two-part divisions of the text up to a limit.

        Args:
            text: The input text to divide

        Yields:
            Tuples of (first_part, second_part) for each possible division point

        Notes:
            - Division points are limited by self.limit if specified
            - Empty input will not yield any divisions
        """
        # Validate input
        if not text:
            return

        # Calculate the maximum division position
        max_pos = min(len(text), self.limit) + 1 if hasattr(self, 'limit') else len(text) + 1

        # Generate all possible divisions
        for pos in range(1, max_pos):
            yield text[:pos], text[pos:]

    def isegment(
            self,
            text: str,
            chunk_size: Optional[int] = None
    ) -> Generator[str, None, None]:
        """Segment text into words using a generator.

        Args:
            text: Text to segment
            chunk_size: Optional chunk size for processing

        Yields:
            Segmented words
        """
        recursion_depth = 0  # Initialize recursion depth counter

        # Validate chunk_size
        if chunk_size is not None and (not isinstance(chunk_size, int) or chunk_size <= 0):
            raise ValueError(f"Invalid chunk_size: {chunk_size}")

        # Use provided chunk_size or default, respecting security limits
        size = chunk_size or self.chunk_size
        max_chunk = SecurityConstants.MAX_CHUNK_SIZE.value
        size = min(size, max_chunk)

        # Define nested recursive search function with memoization
        def search(
                remain: str,
                prev: str
        ) -> Tuple[float, List[str]]:
            """Recursively find optimal word segmentation using Viterbi algorithm.

            Args:
                remain: Remaining text to segment
                prev: Previous word for conditional probability

            Returns:
                Tuple of (log probability score, list of segmented words)

            Raises:
                RuntimeError: If recursion depth or memory limits are exceeded
            """
            nonlocal recursion_depth
            recursion_depth += 1

            # Check recursion depth limit
            if recursion_depth > MAX_RECURSION_DEPTH:
                raise RuntimeError("Maximum recursion depth exceeded")

            # Check memory usage limit
            max_memo = SecurityConstants.MAX_MEMO_SIZE.value
            if len(memo) > max_memo:
                raise RuntimeError("Memory limit exceeded in segmentation")

            # Base case: no text left to segment
            if not remain:
                recursion_depth -= 1
                return 0.0, []

            # Check memoization cache
            key = (remain, prev)
            if key in memo:
                recursion_depth -= 1
                return memo[key]

            # Try all possible first words and find optimal segmentation
            best_score = -math.inf
            best_words = []
            for prefix, suffix in self.divide(remain):
                # Calculate log probability of current word given previous
                score_prefix = math.log10(self.score(prefix, prev))

                # Recursively find best segmentation of remaining text
                segment_score, words = search(suffix, prefix)

                # Calculate total score for this segmentation
                total_score = score_prefix + segment_score

                # Update best segmentation if this one is better
                if total_score > best_score:
                    best_score, best_words = total_score, [prefix] + words

            # Save result in memoization cache
            memo[key] = (best_score, best_words)
            recursion_depth -= 1
            return memo[key]

        # Clean input text
        cleaned = self.clean(text)

        # Process text in chunks to handle large inputs
        prev_chunk = ''
        WORD_HISTORY_SIZE = 5  # Number of words to keep for context between chunks
        MAX_RECURSION_DEPTH = 1000

        # Initialize memoization dictionary
        memo: Dict[Tuple[str, str], Tuple[float, List[str]]] = {}

        # Process text in chunks
        for i in range(0, len(cleaned), size):
            chunk = prev_chunk + cleaned[i:i + size]
            memo.clear()  # Clear memoization cache for each chunk

            try:
                # Find optimal segmentation for current chunk
                _, words = search(chunk, '<s>')  # Use sentence start marker

                # Keep last N words for context in next chunk
                prev_chunk = ''.join(words[-WORD_HISTORY_SIZE:])

                # Yield all words except those kept for context
                for word in words[:-WORD_HISTORY_SIZE]:
                    yield word

            except RuntimeError as e:
                logger.error(f"Error processing chunk {i}: {e}")
                yield f"ERROR_CHUNK_{i}"
                prev_chunk = ''  # Reset context after error

        # Process final remaining words
        try:
            memo.clear()
            _, last_words = search(prev_chunk, '<s>')
            for word in last_words:
                yield word

        except RuntimeError as e:
            logger.error(f"Error processing final chunk: {e}")
            yield "ERROR_FINAL_CHUNK"

    def segment(self, text: str, chunk_size: Optional[int] = None) -> List[str]:
        """Segment text into a list of words.

        Args:
            text: Text to be segmented
            chunk_size: Optional maximum size of chunks to process at once

        Returns:
            List of segmented words

        Raises:
            ValueError: If chunk_size exceeds security limits
            RuntimeError: If model data is not loaded and fails to auto-load
        """
        # Validate chunk_size against security limits
        max_chunk = SecurityConstants.MAX_CHUNK_SIZE.value
        if chunk_size is not None and chunk_size > max_chunk:
            raise ValueError(
                f"chunk_size too large: {chunk_size}. Maximum allowed: {max_chunk}"
            )

        # Auto-load model if not already loaded
        if self.total == 0 or not self.unigrams:
            logger.info("Model not loaded. Auto-loading...")
            self.load()

        # Use the generator method and collect results into a list
        return list(self.isegment(text, chunk_size))

    def segment_and_title(
            self,
            text: str,
            chunk_size: Optional[int] = None
    ) -> List[str]:
        """Segment text into words and convert each word to title case.

        Args:
            text: Text to be segmented
            chunk_size: Optional maximum size of chunks to process at once

        Returns:
            List of segmented words with title case (first letter capitalized)

        Raises:
            ValueError: If chunk_size exceeds security limits
            RuntimeError: If model data is not loaded and fails to auto-load
        """
        # Segment first, then apply title case to each word
        return [word.title() for word in self.segment(text, chunk_size)]

    def set_chunk_size(self, size: int) -> None:
        """Set the default chunk size for text segmentation.

        Args:
            size: Size of text chunks to process at once

        Raises:
            ValueError: If size is not a positive integer or exceeds security limits
        """
        # Validate size is within acceptable range
        max_size = SecurityConstants.MAX_CHUNK_SIZE.value
        if not isinstance(size, int) or size <= 0 or size > max_size:
            raise ValueError(
                f"Chunk size must be a positive integer between 1 and {max_size}, got: {size}"
            )

        # Set the chunk size
        self.chunk_size = size
        logger.debug(f"Chunk size set to {size}")