"""Domain knowledge management and rule extraction.

Central store that combines document retrieval with explicit domain
constraints (business rules, feature exclusions, metric requirements).
Part of the Agentic ML Engineer domain knowledge layer (spec sections 13-14).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from domain.ingestion import DomainDocument, DocumentChunk, ingest_document, ingest_text
from domain.retrieval import DomainIndex, RetrievalResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class DomainConstraint(BaseModel):
    """A machine-readable constraint extracted from domain knowledge.

    Constraints translate business rules into actionable ML guidance:
    which features to include/exclude, which metrics matter, and what
    evaluation criteria to apply.
    """

    rule: str  # human-readable description of the rule
    type: Literal[
        "feature_constraint",
        "metric_constraint",
        "evaluation_constraint",
        "business_rule",
    ]
    severity: Literal["info", "warning", "critical"] = "warning"
    source_document_id: str | None = None
    affected_features: list[str] = Field(default_factory=list)
    affected_metrics: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Rule extraction helpers (deterministic, no LLM)
# ---------------------------------------------------------------------------

# Patterns that signal a constraint in free text.  Each tuple is
# (compiled regex, constraint type, severity).
_RULE_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    # Feature exclusions: "do not use feature X", "exclude column Y"
    (
        re.compile(
            r"(?:do\s+not|don'?t|never|must\s+not|exclude|remove|drop)\s+"
            r"(?:use|include)?\s*(?:the\s+)?(?:feature|column|variable|field)\s+"
            r"[\"']?([A-Za-z_][A-Za-z0-9_]*)[\"']?",
            re.IGNORECASE,
        ),
        "feature_constraint",
        "critical",
    ),
    # Feature requirements: "always include feature X"
    (
        re.compile(
            r"(?:always|must|should|require)\s+(?:use|include)\s+"
            r"(?:the\s+)?(?:feature|column|variable|field)\s+"
            r"[\"']?([A-Za-z_][A-Za-z0-9_]*)[\"']?",
            re.IGNORECASE,
        ),
        "feature_constraint",
        "warning",
    ),
    # Metric directives: "use metric X", "optimise for Y"
    (
        re.compile(
            r"(?:use|optimise|optimize|maximise|maximize|minimise|minimize|prefer)\s+"
            r"(?:the\s+)?(?:metric|measure|score|objective)\s+"
            r"[\"']?([A-Za-z_][A-Za-z0-9_]*)[\"']?",
            re.IGNORECASE,
        ),
        "metric_constraint",
        "warning",
    ),
    # Evaluation constraints: "threshold must be at least X"
    (
        re.compile(
            r"(?:threshold|cutoff|minimum|at\s+least)\s+.*?"
            r"(\d+(?:\.\d+)?(?:\s*%)?)",
            re.IGNORECASE,
        ),
        "evaluation_constraint",
        "warning",
    ),
]


def extract_constraints_from_text(
    text: str,
    *,
    source_document_id: str | None = None,
) -> list[DomainConstraint]:
    """Extract machine-readable constraints from free text using regex patterns.

    This is a deterministic heuristic extractor -- no LLM calls.  It
    recognises common phrasing for feature exclusions, metric directives,
    and evaluation thresholds.

    Parameters
    ----------
    text:
        The text to scan for constraints.
    source_document_id:
        Optional document ID to tag extracted constraints with.

    Returns
    -------
    list[DomainConstraint]:
        Any constraints found (may be empty).
    """
    constraints: list[DomainConstraint] = []
    seen_rules: set[str] = set()

    for pattern, constraint_type, severity in _RULE_PATTERNS:
        for match in pattern.finditer(text):
            # Build a normalised rule string from the full match
            rule_text = match.group(0).strip()
            rule_key = rule_text.lower()

            if rule_key in seen_rules:
                continue
            seen_rules.add(rule_key)

            captured = match.group(1).strip() if match.lastindex else ""

            affected_features: list[str] = []
            affected_metrics: list[str] = []

            if constraint_type == "feature_constraint" and captured:
                affected_features = [captured]
            elif constraint_type == "metric_constraint" and captured:
                affected_metrics = [captured]

            constraints.append(
                DomainConstraint(
                    rule=rule_text,
                    type=constraint_type,  # type: ignore[arg-type]
                    severity=severity,  # type: ignore[arg-type]
                    source_document_id=source_document_id,
                    affected_features=affected_features,
                    affected_metrics=affected_metrics,
                )
            )

    return constraints


# ---------------------------------------------------------------------------
# Knowledge store
# ---------------------------------------------------------------------------


class DomainKnowledgeStore:
    """Central store for domain knowledge -- documents + extracted constraints.

    Combines the ingestion and retrieval layers with an explicit
    constraint registry.  Documents are automatically scanned for
    recognisable rules on ingestion.

    Usage::

        store = DomainKnowledgeStore()
        doc = store.add_document("docs/business_rules.md")
        store.add_constraint(DomainConstraint(
            rule="Exclude PII columns from training",
            type="feature_constraint",
            severity="critical",
            affected_features=["ssn", "email", "phone"],
        ))
        results = store.query("churn prediction features")
        exclusions = store.get_feature_exclusions()
    """

    def __init__(self) -> None:
        self._index = DomainIndex()
        self._constraints: list[DomainConstraint] = []

    # ------------------------------------------------------------------
    # Document management
    # ------------------------------------------------------------------

    def add_document(
        self,
        path_or_text: str | Path,
        source: str | None = None,
        *,
        chunk_size: int = 500,
        overlap: int = 50,
        auto_extract_rules: bool = True,
    ) -> DomainDocument:
        """Ingest a document and add it to the index.

        If *path_or_text* is a path to an existing ``.md`` or ``.txt``
        file it is read from disk; otherwise it is treated as inline text.

        Parameters
        ----------
        path_or_text:
            File path or raw text content.
        source:
            Label for the content origin.  Defaults to the filename
            (for files) or ``"inline"`` (for raw text).
        chunk_size:
            Target characters per chunk.
        overlap:
            Overlap characters between chunks.
        auto_extract_rules:
            When ``True`` (the default), automatically scan the document
            text for recognisable constraints and add them.

        Returns
        -------
        DomainDocument:
            The ingested document.
        """
        path = Path(path_or_text) if not isinstance(path_or_text, Path) else path_or_text

        if path.exists() and path.suffix.lower() in (".md", ".txt"):
            doc = ingest_document(path, chunk_size=chunk_size, overlap=overlap)
            raw_text = path.read_text(encoding="utf-8")
        else:
            # Treat as inline text
            label = source or "inline"
            doc = ingest_text(
                str(path_or_text), source=label, chunk_size=chunk_size, overlap=overlap
            )
            raw_text = str(path_or_text)

        self._index.add_document(doc)

        if auto_extract_rules:
            extracted = extract_constraints_from_text(
                raw_text, source_document_id=doc.document_id
            )
            for constraint in extracted:
                self._constraints.append(constraint)
            if extracted:
                logger.info(
                    "Auto-extracted %d constraint(s) from '%s'.",
                    len(extracted),
                    doc.source,
                )

        return doc

    def remove_document(self, document_id: str) -> bool:
        """Remove a document from the index and its auto-extracted constraints.

        Returns ``True`` if the document was found and removed.
        """
        removed = self._index.remove_document(document_id)
        if removed:
            self._constraints = [
                c
                for c in self._constraints
                if c.source_document_id != document_id
            ]
        return removed

    # ------------------------------------------------------------------
    # Constraint management
    # ------------------------------------------------------------------

    def add_constraint(self, constraint: DomainConstraint) -> None:
        """Manually add a domain constraint."""
        self._constraints.append(constraint)

    def get_constraints(
        self,
        *,
        feature: str | None = None,
        constraint_type: str | None = None,
        severity: str | None = None,
    ) -> list[DomainConstraint]:
        """Get constraints, optionally filtered.

        Parameters
        ----------
        feature:
            If given, return only constraints that mention this feature.
        constraint_type:
            If given, filter by constraint type (e.g. ``"feature_constraint"``).
        severity:
            If given, filter by severity level.

        Returns
        -------
        list[DomainConstraint]:
            Matching constraints (unordered).
        """
        results = self._constraints

        if feature is not None:
            feature_lower = feature.lower()
            results = [
                c
                for c in results
                if any(f.lower() == feature_lower for f in c.affected_features)
            ]

        if constraint_type is not None:
            results = [c for c in results if c.type == constraint_type]

        if severity is not None:
            results = [c for c in results if c.severity == severity]

        return results

    def get_feature_exclusions(self) -> list[str]:
        """Get features that should be excluded based on constraints.

        Scans all ``feature_constraint`` constraints whose rule text
        contains exclusion language (``exclude``, ``do not use``, etc.)
        and returns the union of their ``affected_features``.
        """
        exclusion_keywords = {"exclude", "remove", "drop", "do not", "don't", "never", "must not"}
        excluded: set[str] = set()

        for constraint in self._constraints:
            if constraint.type != "feature_constraint":
                continue
            rule_lower = constraint.rule.lower()
            if any(kw in rule_lower for kw in exclusion_keywords):
                excluded.update(constraint.affected_features)

        return sorted(excluded)

    def get_required_metrics(self) -> list[str]:
        """Get metrics that constraints say should be used.

        Returns the union of ``affected_metrics`` across all
        ``metric_constraint`` entries.
        """
        metrics: set[str] = set()
        for constraint in self._constraints:
            if constraint.type == "metric_constraint":
                metrics.update(constraint.affected_metrics)
        return sorted(metrics)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def query(self, text: str, top_k: int = 5) -> list[RetrievalResult]:
        """Retrieve relevant knowledge chunks for a query.

        Parameters
        ----------
        text:
            Natural-language query.
        top_k:
            Maximum number of chunks to return.

        Returns
        -------
        list[RetrievalResult]:
            Chunks ranked by relevance.
        """
        return self._index.query(text, top_k=top_k)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Summary of all domain knowledge currently held.

        Returns a dict with document counts, chunk counts, constraint
        breakdown, and lists of excluded features / required metrics.
        """
        constraint_by_type: dict[str, int] = {}
        constraint_by_severity: dict[str, int] = {}
        for c in self._constraints:
            constraint_by_type[c.type] = constraint_by_type.get(c.type, 0) + 1
            constraint_by_severity[c.severity] = (
                constraint_by_severity.get(c.severity, 0) + 1
            )

        return {
            "documents": self._index.document_count(),
            "chunks": self._index.chunk_count(),
            "constraints": len(self._constraints),
            "constraints_by_type": constraint_by_type,
            "constraints_by_severity": constraint_by_severity,
            "excluded_features": self.get_feature_exclusions(),
            "required_metrics": self.get_required_metrics(),
            "document_sources": [
                doc.source for doc in self._index.all_documents()
            ],
        }
