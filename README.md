# Python Word Segmentation

A secure, importable implementation of English word segmentation using unigrams and bigrams with stupid backoff algorithm. This is a secure and enhanced version inspired by the original WordSegment library (https://github.com/grantjenks/python-wordsegment/tree/master).

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Overview

`word_segment` is a Python library for splitting concatenated text into individual English words. It's particularly useful for:

- Splitting hashtags (#iloveprogramming → "i love programming")
- Restoring spaces to text with missing spaces (machinelearningisfun → "machine learning is fun")
- Parsing URLs, file paths, or identifiers into human-readable text

The package uses both unigram and bigram frequencies derived from a large corpus of English text, implementing a backoff algorithm for robust segmentation.

## Installation

```bash
pip install -e git@github.com:eleos-health/word_segment.git
```

## Quick Start

```python
from word_segment.word_segment import Segmenter

# Create a segmenter instance
segmenter = Segmenter()

# Load language model data
segmenter.load()

# Basic usage
text = "thisisatest"
print(segmenter.segment(text))
# Output: ['this', 'is', 'a', 'test']

# Title case output
print(segmenter.segment_and_title(text))
# Output: ['This', 'Is', 'A', 'Test']
```

## Features

- **Secure processing**: Validates inputs and limits resource usage
- **Efficient chunked processing**: Handles large inputs by processing text in manageable chunks
- **Title case conversion**: Optional conversion of words to title case
- **Streaming API**: `isegment()` generator for memory-efficient processing of large texts
- **Customizable**: Configurable chunk sizes and model data paths
- **Object-oriented design**: Create multiple segmenter instances with different configurations

## Detailed Usage

### Load with custom data paths

```python
segmenter = Segmenter()
segmenter.load(
    unigrams_path='/path/to/unigrams.txt',
    bigrams_path='/path/to/bigrams.txt',
    words_path='/path/to/words.txt'
)
```

### Process large texts efficiently with chunks

```python
# Create and configure a segmenter
segmenter = Segmenter()
segmenter.load()

# Set default chunk size for all operations
segmenter.set_chunk_size(5000)

# Override chunk size for a specific operation
large_text = "..." # very long text
results = segmenter.segment(large_text, chunk_size=10000)
```

### Use the generator interface for streaming

```python
# Memory-efficient processing of large inputs
segmenter = Segmenter()
segmenter.load()

for word in segmenter.isegment("thisisareallylongpiece..."):
    print(word)
```

### Multiple segmenter instances with different configurations

```python
from word_segment.word_segment import Segmenter

# Create segmenters with different configurations
default_segmenter = Segmenter()
default_segmenter.load()

# Custom segmenter with different chunk size
large_chunk_segmenter = Segmenter()
large_chunk_segmenter.load()
large_chunk_segmenter.set_chunk_size(10000)

# Use them independently
result1 = default_segmenter.segment("thequickbrownfox")
result2 = large_chunk_segmenter.segment("thequickbrownfox")
```

## How It Works

1. The algorithm uses dynamic programming with Viterbi optimization to find the most likely sequence of words.
2. For scoring, it uses conditional probability with bigram context when available, falling back to unigram probabilities.
3. Unknown words are handled with Laplace smoothing.
4. The implementation includes memoization to avoid redundant calculations.

## Security Features

- Path validation to prevent directory traversal
- Input size limits to prevent denial of service
- Memory usage monitoring
- Recursion depth limits
- Protection against malformed data files

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

Original WordSegment library: Copyright 2018 Grant Jenks

## Acknowledgments

This package builds on the work by Grant Jenks on the original WordSegment library, with added security features and optimizations.

Created/edited by ilan.k@eleos.health - Ilan Kahan

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Issues

If you encounter any problems, please file an issue along with a detailed description.