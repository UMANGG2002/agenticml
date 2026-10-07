"""Unit tests for the domain knowledge layer.

Covers:
  - domain.ingestion: ingest_text, ingest_document, chunking logic
  - domain.retrieval: DomainIndex add/query/remove
  - domain.knowledge: DomainKnowledgeStore, constraint extraction, summary
"""

from __future__ import annotations

import pytest

from domain.ingestion import (
    DocumentChunk,
    DomainDocument,
    _chunk_text,
    _generate_chunk_id,
    _generate_document_id,
    _normalise_whitespace,
    ingest_document,
    ingest_text,
)
from domain.retrieval import DomainIndex, RetrievalResult
from domain.knowledge import (
    DomainConstraint,
    DomainKnowledgeStore,
    extract_constraints_from_text,
)


# ======================================================================
# ingestion — ingest_text
# ======================================================================


class TestIngestText:
    """Tests for ingest_text()."""

    def test_returns_domain_document(self):
        doc = ingest_text("Hello world", source="test")
        assert isinstance(doc, DomainDocument)
        assert doc.source == "test"
        assert doc.document_id.startswith("doc_")

    def test_deterministic_document_id(self):
        doc1 = ingest_text("Same content", source="src")
        doc2 = ingest_text("Same content", source="src")
        assert doc1.document_id == doc2.document_id

    def test_different_content_different_id(self):
        doc1 = ingest_text("Content A", source="src")
        doc2 = ingest_text("Content B", source="src")
        assert doc1.document_id != doc2.document_id

    def test_chunk_ids_format(self):
        doc = ingest_text("Short text", source="test")
        assert len(doc.chunks) >= 1
        for i, chunk in enumerate(doc.chunks):
            assert chunk.chunk_id == f"{doc.document_id}_chunk_{i:04d}"

    def test_chunks_contain_text(self):
        doc = ingest_text("Hello world", source="test")
        assert any("Hello" in c.text or "world" in c.text for c in doc.chunks)

    def test_chunk_document_id_matches(self):
        doc = ingest_text("Some text here", source="test")
        for chunk in doc.chunks:
            assert chunk.document_id == doc.document_id

    def test_chunk_source_matches(self):
        doc = ingest_text("Data", source="my_source")
        for chunk in doc.chunks:
            assert chunk.source == "my_source"

    def test_default_source_is_inline(self):
        doc = ingest_text("Data")
        assert doc.source == "inline"

    def test_created_at_is_set(self):
        doc = ingest_text("Data", source="test")
        assert doc.created_at  # non-empty ISO string

    def test_empty_text_produces_no_chunks(self):
        doc = ingest_text("   ", source="test")
        assert doc.chunks == []


# ======================================================================
# ingestion — chunking
# ======================================================================


class TestChunking:
    """Tests for _chunk_text and chunk_size / overlap behaviour."""

    def test_short_text_single_chunk(self):
        chunks = _chunk_text("Short text.", chunk_size=500, overlap=50)
        assert len(chunks) == 1
        assert chunks[0] == "Short text."

    def test_respects_chunk_size(self):
        # Build text that exceeds chunk_size
        text = " ".join(["word"] * 200)  # ~1000 chars
        chunks = _chunk_text(text, chunk_size=100, overlap=0)
        assert len(chunks) > 1
        # First chunk should not exceed chunk_size
        assert len(chunks[0]) <= 100

    def test_overlap_adds_content_from_previous_chunk(self):
        # Two paragraphs, each under chunk_size, but combined over it
        para1 = "A" * 60
        para2 = "B" * 60
        text = f"{para1}\n\n{para2}"
        chunks = _chunk_text(text, chunk_size=80, overlap=20)
        assert len(chunks) >= 2
        # Second chunk should contain some tail of the first chunk
        assert chunks[1].startswith(para1[-20:])

    def test_zero_overlap(self):
        text = "First paragraph.\n\nSecond paragraph."
        chunks = _chunk_text(text, chunk_size=20, overlap=0)
        assert len(chunks) >= 2

    def test_invalid_chunk_size_raises(self):
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            _chunk_text("text", chunk_size=0, overlap=0)

    def test_negative_overlap_raises(self):
        with pytest.raises(ValueError, match="overlap must be non-negative"):
            _chunk_text("text", chunk_size=100, overlap=-1)

    def test_overlap_ge_chunk_size_raises(self):
        with pytest.raises(ValueError, match="overlap must be smaller"):
            _chunk_text("text", chunk_size=100, overlap=100)

    def test_paragraph_boundary_splitting(self):
        para1 = "First paragraph with some text."
        para2 = "Second paragraph with different text."
        text = f"{para1}\n\n{para2}"
        # chunk_size large enough for each paragraph individually
        chunks = _chunk_text(text, chunk_size=50, overlap=0)
        assert len(chunks) == 2
        assert chunks[0] == para1
        assert chunks[1] == para2

    def test_ingest_text_with_custom_chunk_size(self):
        long_text = "word " * 300  # ~1500 chars
        doc = ingest_text(long_text, source="test", chunk_size=100, overlap=10)
        assert len(doc.chunks) > 1
        for chunk in doc.chunks:
            assert chunk.metadata["total_chunks"] == len(doc.chunks)


# ======================================================================
# ingestion — file-based ingestion
# ======================================================================


class TestIngestDocument:
    """Tests for ingest_document() using tmp_path fixture."""

    def test_ingest_markdown_file(self, tmp_path):
        md_file = tmp_path / "doc.md"
        md_file.write_text("# Title\n\nSome content here.", encoding="utf-8")
        doc = ingest_document(md_file)
        assert doc.source == "doc.md"
        assert len(doc.chunks) >= 1

    def test_ingest_txt_file(self, tmp_path):
        txt_file = tmp_path / "notes.txt"
        txt_file.write_text("Plain text notes.", encoding="utf-8")
        doc = ingest_document(txt_file)
        assert doc.source == "notes.txt"
        assert len(doc.chunks) >= 1

    def test_unsupported_extension_raises(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b,c", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported file type"):
            ingest_document(csv_file)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ingest_document(tmp_path / "nonexistent.md")

    def test_markdown_metadata_extraction(self, tmp_path):
        md_file = tmp_path / "headings.md"
        md_file.write_text(
            "# Main Title\n\n## Section A\n\nContent A.\n\n## Section B\n\nContent B.",
            encoding="utf-8",
        )
        doc = ingest_document(md_file)
        # At least one chunk should have headings metadata
        has_headings = any("document_headings" in c.metadata for c in doc.chunks)
        assert has_headings

    def test_source_type_metadata(self, tmp_path):
        md_file = tmp_path / "file.md"
        md_file.write_text("Content", encoding="utf-8")
        doc = ingest_document(md_file)
        assert doc.chunks[0].metadata["source_type"] == "markdown"

        txt_file = tmp_path / "file.txt"
        txt_file.write_text("Content", encoding="utf-8")
        doc_txt = ingest_document(txt_file)
        assert doc_txt.chunks[0].metadata["source_type"] == "text"


# ======================================================================
# retrieval — DomainIndex
# ======================================================================


class TestDomainIndex:
    """Tests for DomainIndex add_document, query, remove_document."""

    def _make_doc(self, text: str, source: str = "test") -> DomainDocument:
        return ingest_text(text, source=source)

    def test_add_document_increases_counts(self):
        idx = DomainIndex()
        assert idx.document_count() == 0
        assert idx.chunk_count() == 0

        doc = self._make_doc("Some interesting content about machine learning.")
        idx.add_document(doc)

        assert idx.document_count() == 1
        assert idx.chunk_count() >= 1

    def test_add_duplicate_document_is_idempotent(self):
        idx = DomainIndex()
        doc = self._make_doc("Content")
        idx.add_document(doc)
        idx.add_document(doc)  # same doc_id
        assert idx.document_count() == 1

    def test_remove_document_returns_true(self):
        idx = DomainIndex()
        doc = self._make_doc("Content to remove")
        idx.add_document(doc)
        assert idx.remove_document(doc.document_id) is True
        assert idx.document_count() == 0
        assert idx.chunk_count() == 0

    def test_remove_nonexistent_returns_false(self):
        idx = DomainIndex()
        assert idx.remove_document("doc_nonexistent") is False

    def test_query_empty_index(self):
        idx = DomainIndex()
        results = idx.query("anything")
        assert results == []

    def test_query_returns_retrieval_results(self):
        idx = DomainIndex()
        doc = self._make_doc(
            "Customer churn prediction is an important business problem. "
            "We use logistic regression and random forests to predict churn."
        )
        idx.add_document(doc)
        results = idx.query("churn prediction")
        assert len(results) >= 1
        assert all(isinstance(r, RetrievalResult) for r in results)
        assert all(r.score > 0 for r in results)

    def test_query_results_sorted_descending(self):
        idx = DomainIndex()
        idx.add_document(
            self._make_doc(
                "Machine learning uses algorithms to learn from data.\n\n"
                "Deep learning is a subset of machine learning using neural networks.",
                source="ml",
            )
        )
        idx.add_document(
            self._make_doc(
                "Cooking pasta requires boiling water and adding salt.",
                source="cooking",
            )
        )
        results = idx.query("machine learning algorithms", top_k=10)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_query_top_k_limits_results(self):
        idx = DomainIndex()
        # Add enough content to get multiple chunks
        for i in range(5):
            idx.add_document(
                self._make_doc(f"Document {i} about topic {i} with unique words_{i}.", source=f"doc{i}")
            )
        results = idx.query("topic", top_k=2)
        assert len(results) <= 2

    def test_all_documents(self):
        idx = DomainIndex()
        doc1 = self._make_doc("First", source="a")
        doc2 = self._make_doc("Second", source="b")
        idx.add_document(doc1)
        idx.add_document(doc2)
        docs = idx.all_documents()
        assert len(docs) == 2
        sources = {d.source for d in docs}
        assert sources == {"a", "b"}

    def test_query_after_remove(self):
        idx = DomainIndex()
        doc = self._make_doc("Unique searchable content about quantum computing.")
        idx.add_document(doc)
        assert len(idx.query("quantum computing")) >= 1

        idx.remove_document(doc.document_id)
        assert idx.query("quantum computing") == []


# ======================================================================
# knowledge — extract_constraints_from_text
# ======================================================================


class TestExtractConstraints:
    """Tests for extract_constraints_from_text()."""

    def test_feature_exclusion_do_not_use(self):
        text = "Do not use feature ssn in the model."
        constraints = extract_constraints_from_text(text)
        assert len(constraints) >= 1
        fc = [c for c in constraints if c.type == "feature_constraint"]
        assert len(fc) >= 1
        assert "ssn" in fc[0].affected_features
        assert fc[0].severity == "critical"

    def test_feature_exclusion_exclude(self):
        text = "Exclude column email from training data."
        constraints = extract_constraints_from_text(text)
        fc = [c for c in constraints if c.type == "feature_constraint"]
        assert len(fc) >= 1
        assert "email" in fc[0].affected_features

    def test_feature_exclusion_never(self):
        text = "Never use feature phone_number."
        constraints = extract_constraints_from_text(text)
        fc = [c for c in constraints if c.type == "feature_constraint"]
        assert len(fc) >= 1
        assert "phone_number" in fc[0].affected_features

    def test_metric_directive(self):
        text = "Optimize the metric AUC_ROC."
        constraints = extract_constraints_from_text(text)
        mc = [c for c in constraints if c.type == "metric_constraint"]
        assert len(mc) >= 1
        assert "AUC_ROC" in mc[0].affected_metrics

    def test_metric_directive_use(self):
        text = "Use the metric F1_score for evaluation."
        constraints = extract_constraints_from_text(text)
        mc = [c for c in constraints if c.type == "metric_constraint"]
        assert len(mc) >= 1
        assert "F1_score" in mc[0].affected_metrics

    def test_evaluation_constraint(self):
        text = "The threshold must be at least 0.85."
        constraints = extract_constraints_from_text(text)
        ec = [c for c in constraints if c.type == "evaluation_constraint"]
        assert len(ec) >= 1

    def test_no_constraints_in_plain_text(self):
        text = "This is just a normal sentence about data science."
        constraints = extract_constraints_from_text(text)
        assert constraints == []

    def test_source_document_id_attached(self):
        text = "Do not use feature age."
        constraints = extract_constraints_from_text(text, source_document_id="doc_abc")
        assert all(c.source_document_id == "doc_abc" for c in constraints)

    def test_no_duplicate_rules(self):
        text = "Do not use feature ssn. Do not use feature ssn."
        constraints = extract_constraints_from_text(text)
        rules = [c.rule for c in constraints]
        # Should deduplicate
        assert len(rules) == len(set(r.lower() for r in rules))

    def test_multiple_constraints_in_one_text(self):
        text = (
            "Do not use feature ssn in any model.\n"
            "Optimize the metric precision.\n"
            "The threshold must be at least 0.90."
        )
        constraints = extract_constraints_from_text(text)
        types = {c.type for c in constraints}
        assert "feature_constraint" in types
        assert "metric_constraint" in types
        assert "evaluation_constraint" in types


# ======================================================================
# knowledge — DomainKnowledgeStore
# ======================================================================


class TestDomainKnowledgeStore:
    """Tests for DomainKnowledgeStore."""

    def test_add_document_inline_text(self):
        store = DomainKnowledgeStore()
        doc = store.add_document("Some domain knowledge.", source="test")
        assert isinstance(doc, DomainDocument)
        assert store.summary()["documents"] == 1

    def test_add_document_from_file(self, tmp_path):
        md_file = tmp_path / "rules.md"
        md_file.write_text("# Business Rules\n\nDo not use feature ssn.", encoding="utf-8")
        store = DomainKnowledgeStore()
        doc = store.add_document(md_file)
        assert doc.source == "rules.md"
        assert store.summary()["documents"] == 1

    def test_auto_constraint_extraction_on_add(self):
        store = DomainKnowledgeStore()
        store.add_document(
            "Do not use feature credit_score. Optimize the metric recall.",
            source="rules",
        )
        summary = store.summary()
        assert summary["constraints"] >= 2
        assert "credit_score" in summary["excluded_features"]
        assert "recall" in summary["required_metrics"]

    def test_auto_extraction_disabled(self):
        store = DomainKnowledgeStore()
        store.add_document(
            "Do not use feature ssn.",
            source="rules",
            auto_extract_rules=False,
        )
        assert store.summary()["constraints"] == 0

    def test_manual_add_constraint(self):
        store = DomainKnowledgeStore()
        store.add_constraint(
            DomainConstraint(
                rule="Exclude PII columns",
                type="feature_constraint",
                severity="critical",
                affected_features=["ssn", "email"],
            )
        )
        constraints = store.get_constraints(constraint_type="feature_constraint")
        assert len(constraints) == 1
        assert "ssn" in constraints[0].affected_features

    def test_get_feature_exclusions(self):
        store = DomainKnowledgeStore()
        store.add_document(
            "Do not use feature ssn. Never use feature email.",
            source="rules",
        )
        exclusions = store.get_feature_exclusions()
        assert "email" in exclusions
        assert "ssn" in exclusions

    def test_get_required_metrics(self):
        store = DomainKnowledgeStore()
        store.add_document(
            "Optimize the metric AUC. Use the metric precision.",
            source="rules",
        )
        metrics = store.get_required_metrics()
        assert "AUC" in metrics
        assert "precision" in metrics

    def test_get_constraints_filter_by_feature(self):
        store = DomainKnowledgeStore()
        store.add_constraint(
            DomainConstraint(
                rule="Exclude ssn",
                type="feature_constraint",
                affected_features=["ssn"],
            )
        )
        store.add_constraint(
            DomainConstraint(
                rule="Exclude email",
                type="feature_constraint",
                affected_features=["email"],
            )
        )
        results = store.get_constraints(feature="ssn")
        assert len(results) == 1
        assert "ssn" in results[0].affected_features

    def test_get_constraints_filter_by_severity(self):
        store = DomainKnowledgeStore()
        store.add_constraint(
            DomainConstraint(
                rule="Info rule", type="business_rule", severity="info"
            )
        )
        store.add_constraint(
            DomainConstraint(
                rule="Critical rule", type="business_rule", severity="critical"
            )
        )
        results = store.get_constraints(severity="critical")
        assert len(results) == 1
        assert results[0].severity == "critical"

    def test_remove_document_removes_constraints(self):
        store = DomainKnowledgeStore()
        doc = store.add_document(
            "Do not use feature ssn.",
            source="rules",
        )
        assert store.summary()["constraints"] >= 1

        store.remove_document(doc.document_id)
        assert store.summary()["documents"] == 0
        assert store.summary()["constraints"] == 0

    def test_remove_nonexistent_document(self):
        store = DomainKnowledgeStore()
        assert store.remove_document("doc_nonexistent") is False

    def test_query_returns_results(self):
        store = DomainKnowledgeStore()
        store.add_document(
            "Customer churn prediction uses historical transaction data.",
            source="churn_doc",
        )
        results = store.query("churn prediction")
        assert len(results) >= 1
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_query_empty_store(self):
        store = DomainKnowledgeStore()
        assert store.query("anything") == []


# ======================================================================
# knowledge — summary
# ======================================================================


class TestSummary:
    """Tests for DomainKnowledgeStore.summary()."""

    def test_summary_empty_store(self):
        store = DomainKnowledgeStore()
        s = store.summary()
        assert s["documents"] == 0
        assert s["chunks"] == 0
        assert s["constraints"] == 0
        assert s["excluded_features"] == []
        assert s["required_metrics"] == []
        assert s["document_sources"] == []

    def test_summary_counts_correct(self):
        store = DomainKnowledgeStore()
        store.add_document("First document about topic A.", source="doc_a")
        store.add_document("Second document about topic B.", source="doc_b")
        store.add_constraint(
            DomainConstraint(rule="rule1", type="business_rule", severity="info")
        )

        s = store.summary()
        assert s["documents"] == 2
        assert s["chunks"] >= 2
        assert s["constraints"] >= 1
        assert set(s["document_sources"]) == {"doc_a", "doc_b"}

    def test_summary_constraints_by_type(self):
        store = DomainKnowledgeStore()
        store.add_constraint(
            DomainConstraint(rule="r1", type="feature_constraint")
        )
        store.add_constraint(
            DomainConstraint(rule="r2", type="metric_constraint")
        )
        store.add_constraint(
            DomainConstraint(rule="r3", type="feature_constraint")
        )

        s = store.summary()
        assert s["constraints_by_type"]["feature_constraint"] == 2
        assert s["constraints_by_type"]["metric_constraint"] == 1

    def test_summary_constraints_by_severity(self):
        store = DomainKnowledgeStore()
        store.add_constraint(
            DomainConstraint(rule="r1", type="business_rule", severity="critical")
        )
        store.add_constraint(
            DomainConstraint(rule="r2", type="business_rule", severity="warning")
        )

        s = store.summary()
        assert s["constraints_by_severity"]["critical"] == 1
        assert s["constraints_by_severity"]["warning"] == 1


# ======================================================================
# ingestion — helper functions
# ======================================================================


class TestHelpers:
    """Tests for internal ingestion helpers."""

    def test_generate_document_id_deterministic(self):
        id1 = _generate_document_id("text", "src")
        id2 = _generate_document_id("text", "src")
        assert id1 == id2
        assert id1.startswith("doc_")

    def test_generate_chunk_id_format(self):
        cid = _generate_chunk_id("doc_abc", 0)
        assert cid == "doc_abc_chunk_0000"
        cid = _generate_chunk_id("doc_abc", 42)
        assert cid == "doc_abc_chunk_0042"

    def test_normalise_whitespace(self):
        assert _normalise_whitespace("  hello   world  ") == "hello world"
        # Preserves paragraph breaks
        result = _normalise_whitespace("para1\n\npara2")
        assert "\n\n" in result

    def test_normalise_collapses_excessive_newlines(self):
        result = _normalise_whitespace("a\n\n\n\n\nb")
        assert result == "a\n\nb"
