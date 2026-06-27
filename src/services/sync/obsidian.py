import io
import re
import zipfile
from typing import List
from src.models.paper import Paper

def format_obsidian_markdown(paper: Paper) -> str:
    """Format paper data as Obsidian Markdown with YAML frontmatter."""
    title_clean = paper.title.replace('"', '\\"')
    authors_clean = paper.authors.replace('"', '\\"') if paper.authors else ""
    date_str = paper.published_date.strftime("%Y-%m-%d") if paper.published_date else "Unknown"
    
    categories_list = paper.categories if isinstance(paper.categories, list) else [str(paper.categories)]
    categories_yaml = "\n".join([f"  - {cat}" for cat in categories_list])
    
    # Generate standard Obsidian tag list
    tags_yaml = "\n".join([f"  - research-paper"] + [f"  - {cat.replace('.', '-')}" for cat in categories_list])

    markdown = f"""---
title: "{title_clean}"
arxiv_id: "{paper.arxiv_id}"
authors: "{authors_clean}"
published_date: "{date_str}"
pdf_url: "{paper.pdf_url}"
categories:
{categories_yaml}
tags:
{tags_yaml}
---

# {paper.title}

> **Authors:** {paper.authors}  
> **Published:** {date_str} | **arXiv:** [{paper.arxiv_id}]({paper.pdf_url})

## Abstract
{paper.abstract}

## Notes
*Write your reading notes and insights here...*

## Backlinks
- [[{paper.arxiv_id}]]
"""
    return markdown


def export_collection_as_obsidian_zip(collection_name: str, papers: List[Paper]) -> bytes:
    """Generate a zip archive of Obsidian markdown files for papers in the collection."""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        for paper in papers:
            # Create a clean file name from the paper title
            clean_title = re.sub(r'[\\/*?:"<>|]', "", paper.title)
            clean_title = clean_title.replace(" ", "_")[:60]
            filename = f"{clean_title}_{paper.arxiv_id}.md"
            
            content = format_obsidian_markdown(paper)
            zip_file.writestr(filename, content)
            
    return zip_buffer.getvalue()
