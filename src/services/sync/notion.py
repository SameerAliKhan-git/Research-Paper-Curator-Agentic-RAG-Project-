import logging
from typing import List, Dict, Any, Optional
import httpx
from src.models.paper import Paper

logger = logging.getLogger(__name__)


class NotionSyncService:
    """Service to synchronize research collections to Notion databases using Notion REST API."""

    def __init__(self, auth_token: Optional[str] = None):
        self.auth_token = auth_token
        self.headers = {
            "Authorization": f"Bearer {auth_token}" if auth_token else "",
            "Content-Type": "application/json",
            "Notion-Version": "2022-06-28",
        }

    async def sync_paper_to_database(self, database_id: str, paper: Paper) -> bool:
        """Create a new page in a Notion database containing paper metadata."""
        if not self.auth_token:
            logger.info("Notion auth token missing. Mocking Notion sync for paper: %s", paper.arxiv_id)
            return True

        url = "https://api.notion.com/v1/pages"
        
        # Build Notion page properties
        payload = {
            "parent": {"database_id": database_id},
            "properties": {
                "Name": {
                    "title": [
                        {
                            "text": {
                                "content": paper.title[:2000]  # Notion limit
                            }
                        }
                    ]
                },
                "arXiv ID": {
                    "rich_text": [
                        {
                            "text": {
                                "content": paper.arxiv_id
                            }
                        }
                    ]
                },
                "Authors": {
                    "rich_text": [
                        {
                            "text": {
                                "content": (paper.authors or "")[:2000]
                            }
                        }
                    ]
                },
                "URL": {
                    "url": paper.pdf_url or f"https://arxiv.org/abs/{paper.arxiv_id}"
                },
                "Categories": {
                    "multi_select": [
                        {"name": cat} for cat in (paper.categories if isinstance(paper.categories, list) else [str(paper.categories)])
                    ]
                }
            },
            "children": [
                {
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {
                        "rich_text": [{"type": "text", "text": {"content": "Abstract"}}]
                    }
                },
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"type": "text", "text": {"content": paper.abstract[:2000]}}]
                    }
                }
            ]
        }

        try:
            async with httpx.AsyncClient() as client:
                res = await client.post(url, json=payload, headers=self.headers, timeout=10.0)
                if res.status_code == 200:
                    logger.info("Successfully synced paper %s to Notion", paper.arxiv_id)
                    return True
                else:
                    logger.error("Notion API error (%d): %s", res.status_code, res.text)
                    return False
        except Exception as e:
            logger.error("Failed to sync paper %s to Notion: %s", paper.arxiv_id, e)
            return False

    async def sync_collection(self, database_id: str, papers: List[Paper]) -> Dict[str, Any]:
        """Synchronize all papers in a collection to Notion."""
        success_count = 0
        failed_count = 0
        
        for paper in papers:
            success = await self.sync_paper_to_database(database_id, paper)
            if success:
                success_count += 1
            else:
                failed_count += 1
                
        return {
            "total_papers": len(papers),
            "synced_successfully": success_count,
            "failed_sync": failed_count,
            "mocked": not bool(self.auth_token)
        }
