"""Domain knowledge retrieval using TF-IDF.

Provides an in-memory search index over ingested document chunks.
No external vector database required -- uses scikit-learn's
TfidfVectorizer for the MVP. Part of the Agentic ML Engineer
domain knowledge layer (spec sections 13-14).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from domain.ingestion import DocumentChunk, DomainDocument

logger = logging.getLogger(__name__)


class RetrievalResult(BaseModel):
    """A single retrieval hit with its relevance score."""

    chunk: DocumentChunk
    score: float


class DomainIndex:
    """In-memory TF-IDF index over domain document chunks.

    Usage::

        index = DomainIndex()
        index.add_document(doc)
        results = index.query("customer churn definition", top_k=3)
    """

    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer(
            strip_accents="unicode",
            lowercase=True,
            stop_words="english",
            max_df=0.95,
            min_df=1,
            ngram_range=(1, 2),
        )
        self._chunks: list[DocumentChunk] = []
        self._documents: dict[str, DomainDocument] = {}  # doc_id -> document
        self._tfidf_matrix: Any | None = None  # sparse CSR matrix
        self._dirty: bool = False  # True when chunks added but matrix not rebuilt

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_document(self, doc: DomainDocument) -> None:
        """Add all chunks of a document to the index.

        Parameters
        ----------
        doc:
            A parsed :class:`DomainDocument` (produced by the ingestion layer).
        """
        if doc.document_id in self._documents:
            logger.info(
                "Document '%s' (%s) already indexed -- skipping.",
                doc.document_id,
                doc.source,
            )
            return

        self._documents[doc.document_id] = doc
        self._chunks.extend(doc.chunks)
        self._dirty = True
        logger.info(
            "Indexed %d chunks from '%s' (%s).",
            len(doc.chunks),
            doc.source,
            doc.document_id,
        )

    def remove_document(self, document_id: str) -> bool:
        """Remove a document and its chunks from the index.

        Returns ``True`` if the document was found and removed.
        """
        if document_id not in self._documents:
            return False

        del self._documents[document_id]
        self._chunks = [c for c in self._chunks if c.document_id != document_id]
        self._dirty = True
        return True

    def query(self, text: str, top_k: int = 5) -> list[RetrievalResult]:
        """Retrieve the most relevant chunks for a query string.

        Parameters
        ----------
        text:
            Natural-language query.
        top_k:
            Maximum number of results to return.

        Returns
        -------
        list[RetrievalResult]:
            Results ordered by descending relevance score.
        """
        if not self._chunks:
            return []

        self._rebuild_if_dirty()
        assert self._tfidf_matrix is not None

        query_vec = self._vectorizer.transform([text])
        similarities = cosine_similarity(query_vec, self._tfidf_matrix).flatten()

        # Get the top-k indices (descending score)
        if top_k >= len(similarities):
            top_indices = np.argsort(similarities)[::-1]
        else:
            # Use argpartition for efficiency on large indices
            partitioned = np.argpartition(similarities, -top_k)[-top_k:]
            top_indices = partitioned[np.argsort(similarities[partitioned])[::-1]]

        results: list[RetrievalResult] = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score <= 0.0:
                break  # remaining scores are zero or negative -- stop
            results.append(
                RetrievalResult(chunk=self._chunks[idx], score=round(score, 6))
            )

        return results

    def all_documents(self) -> list[DomainDocument]:
        """Return all indexed documents."""
        return list(self._documents.values())

    def document_count(self) -> int:
        """Number of distinct documents in the index."""
        return len(self._documents)

    def chunk_count(self) -> int:
        """Total number of chunks across all documents."""
        return len(self._chunks)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebuild_if_dirty(self) -> None:
        """Re-fit the TF-IDF matrix when chunks have changed."""
        if not self._dirty:
            return

        corpus = [chunk.text for chunk in self._chunks]
        if corpus:
            # Adjust max_df for small corpora to avoid the
            # "max_df corresponds to < documents than min_df" error
            n = len(corpus)
            self._vectorizer.max_df = max(0.95, (n - 0.5) / n) if n > 1 else 1.0
            self._tfidf_matrix = self._vectorizer.fit_transform(corpus)
        else:
            self._tfidf_matrix = None

        self._dirty = False
        logger.debug("Rebuilt TF-IDF matrix with %d chunks.", len(corpus))
