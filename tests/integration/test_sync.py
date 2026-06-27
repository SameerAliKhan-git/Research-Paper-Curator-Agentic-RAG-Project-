import pytest
from src.models.paper import Paper
from src.services.sync.obsidian import format_obsidian_markdown, export_collection_as_obsidian_zip


def test_obsidian_markdown_formatting():
    paper = Paper(
        arxiv_id="2406.12345",
        title="Attention is All You Need",
        authors="Ashish Vaswani, Noam Shazeer",
        abstract="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...",
        pdf_url="https://arxiv.org/pdf/1706.03762.pdf",
        categories=["cs.CL"]
    )

    md = format_obsidian_markdown(paper)

    assert "arxiv_id: \"2406.12345\"" in md
    assert "title: \"Attention is All You Need\"" in md
    assert "authors: \"Ashish Vaswani, Noam Shazeer\"" in md
    assert "tags:" in md
    assert "- research-paper" in md
    assert "- cs-CL" in md
    assert "## Abstract" in md


def test_obsidian_zip_export():
    papers = [
        Paper(
            arxiv_id="1706.03762",
            title="Attention is All You Need",
            authors="Ashish Vaswani",
            abstract="Abstract 1",
            pdf_url="https://arxiv.org/pdf/1706.03762.pdf",
            categories=["cs.CL"]
        ),
        Paper(
            arxiv_id="1810.04805",
            title="BERT: Pre-training of Deep Bidirectional Transformers",
            authors="Jacob Devlin",
            abstract="Abstract 2",
            pdf_url="https://arxiv.org/pdf/1810.04805.pdf",
            categories=["cs.CL"]
        )
    ]

    zip_bytes = export_collection_as_obsidian_zip("My Deep Learning Collection", papers)
    
    assert zip_bytes is not None
    assert len(zip_bytes) > 0
    # Basic check for zip file magic bytes
    assert zip_bytes.startswith(b"PK\x03\x04")
