"""Tests for the TextChunker service."""

import pytest

from src.services.indexing.text_chunker import TextChunker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def chunker():
    return TextChunker(chunk_size=10, overlap_size=3, min_chunk_size=5)


@pytest.fixture
def large_chunker():
    return TextChunker(chunk_size=600, overlap_size=100, min_chunk_size=100)


# ---------------------------------------------------------------------------
# Tests – chunk respects max size
# ---------------------------------------------------------------------------

class TestChunkRespectsMaxSize:
    def test_chunk_respects_max_size(self, chunker: TextChunker):
        """Each chunk should not exceed chunk_size words."""
        text = " ".join(f"word{i}" for i in range(50))
        chunks = chunker.chunk_text(text, arxiv_id="2301.00001", paper_id="p1")

        for chunk in chunks:
            word_count = len(chunk.text.split())
            # Allow last chunk to be smaller, but none should exceed chunk_size + overlap
            assert word_count <= chunker.chunk_size + chunker.overlap_size

    def test_small_text_produces_single_chunk(self, chunker: TextChunker):
        text = " ".join(f"w{i}" for i in range(8))
        chunks = chunker.chunk_text(text, arxiv_id="2301.00001", paper_id="p1")
        assert len(chunks) == 1

    def test_empty_text_returns_empty(self, chunker: TextChunker):
        chunks = chunker.chunk_text("", arxiv_id="2301.00001", paper_id="p1")
        assert chunks == []


# ---------------------------------------------------------------------------
# Tests – chunk overlap
# ---------------------------------------------------------------------------

class TestChunkOverlap:
    def test_chunk_overlap(self, chunker: TextChunker):
        """Consecutive chunks should share overlap_size words."""
        text = " ".join(f"word{i}" for i in range(30))
        chunks = chunker.chunk_text(text, arxiv_id="2301.00001", paper_id="p1")

        if len(chunks) < 2:
            pytest.skip("Need at least 2 chunks for overlap test")

        for i in range(1, len(chunks)):
            prev_words = chunks[i - 1].text.split()
            curr_words = chunks[i].text.split()
            # Last overlap_size words of previous chunk == first overlap_size words of next chunk
            overlap = prev_words[-chunker.overlap_size:]
            assert curr_words[: chunker.overlap_size] == overlap

    def test_overlap_metadata_recorded(self, chunker: TextChunker):
        text = " ".join(f"word{i}" for i in range(30))
        chunks = chunker.chunk_text(text, arxiv_id="2301.00001", paper_id="p1")

        assert len(chunks) >= 2
        # First chunk has no overlap with previous
        assert chunks[0].metadata.overlap_with_previous == 0
        # Second chunk should record overlap
        assert chunks[1].metadata.overlap_with_previous == chunker.overlap_size


# ---------------------------------------------------------------------------
# Tests – section-based chunking
# ---------------------------------------------------------------------------

class TestSectionBasedChunking:
    def test_section_based_chunking(self, large_chunker: TextChunker):
        """chunk_paper with sections should produce section-titled chunks."""
        sections = {
            "Introduction": " ".join(f"intro{i}" for i in range(200)),
            "Methods": " ".join(f"method{i}" for i in range(500)),
            "Conclusion": " ".join(f"conclusion{i}" for i in range(150)),
        }
        chunks = large_chunker.chunk_paper(
            title="Test Paper",
            abstract="This is a test abstract about testing.",
            full_text="Full text placeholder.",
            arxiv_id="2301.00001",
            paper_id="p1",
            sections=sections,
        )
        assert len(chunks) > 0
        # At least some chunks should have section_title metadata
        section_titles = [c.metadata.section_title for c in chunks if c.metadata.section_title]
        assert len(section_titles) > 0

    def test_section_based_chunking_with_json_string(self, large_chunker: TextChunker):
        """Sections can be passed as a JSON string."""
        import json
        sections = json.dumps({
            "Intro": " ".join(f"word{i}" for i in range(200)),
            "Body": " ".join(f"word{i}" for i in range(300)),
        })
        chunks = large_chunker.chunk_paper(
            title="Test Paper",
            abstract="Test abstract.",
            full_text="Full text.",
            arxiv_id="2301.00001",
            paper_id="p1",
            sections=sections,
        )
        assert len(chunks) > 0

    def test_section_based_chunking_with_list(self, large_chunker: TextChunker):
        """Sections can be passed as a list of dicts."""
        sections = [
            {"title": "Introduction", "content": " ".join(f"intro{i}" for i in range(200))},
            {"title": "Methods", "content": " ".join(f"method{i}" for i in range(300))},
        ]
        chunks = large_chunker.chunk_paper(
            title="Test Paper",
            abstract="Abstract text.",
            full_text="Full text.",
            arxiv_id="2301.00001",
            paper_id="p1",
            sections=sections,
        )
        assert len(chunks) > 0


# ---------------------------------------------------------------------------
# Tests – edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_chunk_text_min_size_fallback(self):
        """Text shorter than min_chunk_size still returns one chunk."""
        tc = TextChunker(chunk_size=100, overlap_size=20, min_chunk_size=50)
        text = " ".join(f"w{i}" for i in range(30))
        chunks = tc.chunk_text(text, arxiv_id="2301.00001", paper_id="p1")
        assert len(chunks) == 1
        assert len(chunks[0].text.split()) == 30

    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError, match="Overlap size must be less than chunk size"):
            TextChunker(chunk_size=100, overlap_size=100)
