from enum import Enum
from pathlib import Path


class DirectoryConstants(Enum):
    MODULE_FOLDER_PATH = Path(__file__).parent.parent.absolute()
    WORD_SEGMENT_PATH = Path(__file__).parent.absolute()

    UNIGRAMS_FILE_PATH = WORD_SEGMENT_PATH / 'unigrams.txt'
    BIGRAMS_FILE_PATH = WORD_SEGMENT_PATH / 'bigrams.txt'
    WORDS_FILE_PATH = WORD_SEGMENT_PATH / 'words.txt'

    SAFE_PATHS = [
        UNIGRAMS_FILE_PATH,
        BIGRAMS_FILE_PATH,
        WORDS_FILE_PATH
    ]


class SecurityConstants(Enum):
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB per file
    MAX_MEMO_SIZE = 100_000  # Maximum entries in memoization dictionary
    MAX_TEXT_LENGTH = 1_000_000  # Maximum length of input text to process
    MAX_LINE_LENGTH = 10_000  # Maximum line length in data files
    DEFAULT_CHUNK_SIZE = 250  # Default chunk size for processing
    MAX_CHUNK_SIZE = 10_000  # Hard upper bound on chunk size
    MAX_COUNT_ENTRIES = 1_000_000  # Maximum entries in counts files


class SegmenterConstants(Enum):
    TOTAL = 1024908267229.0
    LIMIT = 24
    ALPHABET = set('abcdefghijklmnopqrstuvwxyz0123456789')