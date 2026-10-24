"""In-process Okapi BM25 lexical index for hybrid retrieval.

The lexical corpus is kept in memory (rebuilt from the knowledge store on demand) so lexical
search works with or without Postgres, and so indexing stays incremental during ingestion.
"""

import logging
import math
import re
import threading
import unicodedata
from collections import Counter
from heapq import nlargest
from typing import Iterable, Sequence

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)


def strip_diacritics(text: str) -> str:
    """Remove combining marks so accented Vietnamese terms match their base form."""
    decomposed = unicodedata.normalize("NFD", text or "")
    base = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    # Vietnamese tone marks are also written as precomposed characters in some corpora
    return base.replace("đ", "d").replace("Đ", "d")


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase, accent-insensitive terms.

    Args:
        text: Raw text (any language).

    Returns:
        list[str]: Token terms; alphanumeric runs only.
    """
    normalized = strip_diacritics(text).casefold()
    return _TOKEN_RE.findall(normalized)


class BM25Index:
    """Okapi BM25 index supporting incremental add/remove and top-k scoring."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        """Create an empty index.

        Args:
            k1: Term-frequency saturation parameter.
            b: Length-normalization parameter in [0, 1].
        """
        self.k1 = k1
        self.b = b
        self._lock = threading.RLock()
        self._postings: dict[str, dict[str, int]] = {}
        self._doc_tokens: dict[str, list[str]] = {}
        self._doc_length: dict[str, int] = {}
        self._df: dict[str, int] = {}
        self._total_length = 0

    # ------------------------------------------------------------------ mutation
    def clear(self) -> None:
        """Remove every document from the index."""
        with self._lock:
            self._postings.clear()
            self._doc_tokens.clear()
            self._doc_length.clear()
            self._df.clear()
            self._total_length = 0

    def add(self, doc_id: str, text: str) -> None:
        """Add or replace one document, updating document frequencies incrementally."""
        with self._lock:
            if doc_id in self._doc_tokens:
                self.remove(doc_id)
            tokens = tokenize(text)
            if not tokens:
                # Empty documents are tracked without postings so ids remain known
                self._doc_tokens[doc_id] = []
                self._doc_length[doc_id] = 0
                return
            counts = Counter(tokens)
            for term, count in counts.items():
                self._postings.setdefault(term, {})[doc_id] = count
                self._df[term] = self._df.get(term, 0) + 1
            self._doc_tokens[doc_id] = tokens
            self._doc_length[doc_id] = len(tokens)
            self._total_length += len(tokens)

    def add_many(self, documents: Iterable[tuple[str, str]]) -> None:
        """Add many ``(doc_id, text)`` pairs to the index."""
        for doc_id, text in documents:
            self.add(doc_id, text)

    def remove(self, doc_id: str) -> None:
        """Delete a document from the index (no-op when absent)."""
        with self._lock:
            tokens = self._doc_tokens.pop(doc_id, None)
            if tokens is None:
                return
            length = self._doc_length.pop(doc_id, 0)
            self._total_length = max(0, self._total_length - length)
            for term in set(tokens):
                postings = self._postings.get(term)
                if postings is None:
                    continue
                postings.pop(doc_id, None)
                self._df[term] = self._df.get(term, 1) - 1
                if self._df[term] <= 0:
                    self._df.pop(term, None)
                    self._postings.pop(term, None)

    # ------------------------------------------------------------------ introspection
    def __len__(self) -> int:
        """Number of indexed documents."""
        with self._lock:
            return len(self._doc_tokens)

    @property
    def is_empty(self) -> bool:
        """True when no documents with tokens are indexed."""
        with self._lock:
            return not self._postings

    @property
    def avg_doc_length(self) -> float:
        """Average token length of indexed documents."""
        with self._lock:
            count = len(self._doc_length)
            return (self._total_length / count) if count else 0.0

    def document_count(self) -> int:
        """Number of indexed documents (public accessor for API payloads)."""
        return len(self)

    def stats(self) -> dict[str, object]:
        """Index size metrics for diagnostics."""
        with self._lock:
            return {
                "documents": len(self._doc_tokens),
                "terms": len(self._postings),
                "total_tokens": self._total_length,
                "avg_doc_length": round(self.avg_doc_length, 2),
                "k1": self.k1,
                "b": self.b,
            }

    # ------------------------------------------------------------------ scoring
    def _idf(self, term: str, total_docs: int) -> float:
        """Robertson-Sparck Jones IDF with the non-negative BM25+ style shift."""
        df = self._df.get(term, 0)
        if df <= 0:
            return 0.0
        return math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5))

    def score(self, doc_id: str, query_tokens: Sequence[str]) -> float:
        """BM25 score of one document against pre-tokenized query terms."""
        with self._lock:
            total_docs = len(self._doc_tokens)
            if total_docs == 0:
                return 0.0
            length = self._doc_length.get(doc_id, 0)
            if length == 0:
                return 0.0
            postings = self._postings
            avgdl = self.avg_doc_length or 1.0
            doc_postings = [postings.get(term, {}).get(doc_id, 0) for term in query_tokens]
            score = 0.0
            for term, tf in zip(query_tokens, doc_postings):
                if tf <= 0:
                    continue
                idf = self._idf(term, total_docs)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (length / avgdl))
                score += idf * (tf * (self.k1 + 1.0)) / denominator
            return score

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Rank documents against a raw query string.

        Args:
            query: Query text.
            k: Maximum number of results.

        Returns:
            list[tuple[str, float]]: ``(doc_id, score)`` pairs sorted by descending score.
        """
        with self._lock:
            if self.is_empty or k <= 0:
                return []
            tokens = tokenize(query)
            if not tokens:
                return []
            unique_tokens = list(dict.fromkeys(tokens))
            # Candidate pruning: only documents containing at least one query term
            candidates: set[str] = set()
            for term in unique_tokens:
                postings = self._postings.get(term)
                if postings:
                    candidates.update(postings.keys())
            if not candidates:
                return []
            scored = [(doc_id, self.score(doc_id, unique_tokens)) for doc_id in candidates]
            ranked = nlargest(k, scored, key=lambda item: item[1])
            return [(doc_id, score) for doc_id, score in ranked if score > 0.0]


bm25_index = BM25Index(k1=1.5, b=0.75)
