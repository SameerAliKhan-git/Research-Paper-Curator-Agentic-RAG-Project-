import logging
import json
import os
from typing import Dict, List, Any, Optional
from sqlalchemy import text
from src.config import Settings
from src.services.ollama.client import OllamaClient

logger = logging.getLogger(__name__)

class KnowledgeGraphService:
    """GraphRAG Knowledge Graph Service using SQL database for high-performance node/edge storage."""

    def __init__(self, db_interface, settings: Settings, ollama_client: OllamaClient):
        self.db = db_interface
        self.settings = settings
        self.ollama = ollama_client
        self._init_db_tables()

    def _init_db_tables(self):
        """Ensure knowledge graph tables exist in the database."""
        try:
            with self.db.get_session() as session:
                # Use raw SQL to create table if not exists for maximum database portability
                create_table_sql = """
                CREATE TABLE IF NOT EXISTS knowledge_graph_edges (
                    id SERIAL PRIMARY KEY,
                    source VARCHAR(255) NOT NULL,
                    relation VARCHAR(255) NOT NULL,
                    target VARCHAR(255) NOT NULL,
                    arxiv_id VARCHAR(50),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_kg_source ON knowledge_graph_edges(source);
                CREATE INDEX IF NOT EXISTS idx_kg_target ON knowledge_graph_edges(target);
                CREATE INDEX IF NOT EXISTS idx_kg_arxiv ON knowledge_graph_edges(arxiv_id);
                """
                session.execute(text(create_table_sql))
                session.commit()
                logger.info("Knowledge Graph DB tables verified/created successfully.")
        except Exception as e:
            logger.warning(f"Failed to auto-initialize Knowledge Graph DB tables (might be in sqlite/test env): {e}")
            # Try SQLite fallback syntax
            try:
                with self.db.get_session() as session:
                    create_sqlite_sql = """
                    CREATE TABLE IF NOT EXISTS knowledge_graph_edges (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        source TEXT NOT NULL,
                        relation TEXT NOT NULL,
                        target TEXT NOT NULL,
                        arxiv_id TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE INDEX IF NOT EXISTS idx_kg_source ON knowledge_graph_edges(source);
                    CREATE INDEX IF NOT EXISTS idx_kg_target ON knowledge_graph_edges(target);
                    """
                    session.execute(text(create_sqlite_sql))
                    session.commit()
                    logger.info("Knowledge Graph SQLite fallback tables verified/created successfully.")
            except Exception as ex:
                logger.error(f"Failed to initialize Knowledge Graph tables in all dialects: {ex}")

    async def extract_relations_from_paper(self, arxiv_id: str, title: str, abstract: str) -> int:
        """Use LLM to extract entities and relations from a paper abstract and save to KG."""
        prompt = (
            "You are an information extraction assistant. Read the scientific paper abstract "
            "and extract up to 5 key semantic relations between concepts, technologies, datasets, or models.\n\n"
            f"Title: {title}\n"
            f"Abstract: {abstract}\n\n"
            "You must return ONLY a JSON list of objects with keys: 'source', 'relation', 'target'.\n"
            "Keep concept names concise (1-3 words, e.g. 'Transformer', 'Attention mechanism', 'Machine translation').\n"
            "Example output:\n"
            "[\n"
            "  {\"source\": \"Transformer\", \"relation\": \"uses\", \"target\": \"Attention mechanism\"},\n"
            "  {\"source\": \"BERT\", \"relation\": \"trained on\", \"target\": \"Wikipedia dataset\"}\n"
            "]"
        )

        try:
            model_name = self.settings.ollama_model
            response = await self.ollama.generate(model=model_name, prompt=prompt)
            if not response or "response" not in response:
                return 0

            raw_text = response["response"].strip()
            # Clean JSON markdown blocks if present
            if raw_text.startswith("```json"):
                raw_text = raw_text.replace("```json", "", 1)
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3].strip()
            
            relations = json.loads(raw_text)
            inserted = 0

            with self.db.get_session() as session:
                for rel in relations:
                    source = rel.get("source", "").strip()
                    relation = rel.get("relation", "").strip()
                    target = rel.get("target", "").strip()
                    
                    if source and relation and target:
                        insert_sql = """
                        INSERT INTO knowledge_graph_edges (source, relation, target, arxiv_id)
                        VALUES (:source, :relation, :target, :arxiv_id)
                        """
                        session.execute(
                            text(insert_sql),
                            {"source": source, "relation": relation, "target": target, "arxiv_id": arxiv_id}
                        )
                        inserted += 1
                session.commit()
                
            logger.info(f"Extracted and saved {inserted} KG relations for paper {arxiv_id}")
            return inserted

        except Exception as e:
            logger.error(f"Failed to extract KG relations from paper {arxiv_id}: {e}")
            return 0

    async def get_subgraph_for_concepts(self, concepts: List[str], depth: int = 1) -> List[Dict[str, str]]:
        """Retrieve concept relations paths (1-hop or 2-hop neighbor connections) to augment context."""
        if not concepts:
            return []

        results = []
        try:
            with self.db.get_session() as session:
                query = """
                SELECT source, relation, target, arxiv_id
                FROM knowledge_graph_edges
                WHERE LOWER(source) IN :concepts OR LOWER(target) IN :concepts
                LIMIT 50
                """
                # Lowercase clean search terms
                search_terms = tuple(c.strip().lower() for c in concepts)
                rows = session.execute(text(query), {"concepts": search_terms}).fetchall()
                
                for r in rows:
                    results.append({
                        "source": r[0],
                        "relation": r[1],
                        "target": r[2],
                        "arxiv_id": r[3]
                    })
        except Exception as e:
            logger.error(f"Failed to query concept subgraph: {e}")
            
        return results

    async def get_full_graph_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Return full list of unique nodes and edges in the graph for frontend visualization."""
        nodes = []
        edges = []
        seen_nodes = set()
        
        try:
            with self.db.get_session() as session:
                query = "SELECT source, relation, target, arxiv_id FROM knowledge_graph_edges LIMIT 200"
                rows = session.execute(text(query)).fetchall()
                
                for r in rows:
                    src, rel, tgt, paper = r[0], r[1], r[2], r[3]
                    
                    if src not in seen_nodes:
                        seen_nodes.add(src)
                        nodes.append({"id": src, "label": src, "group": "concept"})
                    if tgt not in seen_nodes:
                        seen_nodes.add(tgt)
                        nodes.append({"id": tgt, "label": tgt, "group": "concept"})
                        
                    edges.append({
                        "from": src,
                        "to": tgt,
                        "label": rel,
                        "arxiv_id": paper
                    })
        except Exception as e:
            logger.error(f"Failed to fetch full graph data: {e}")

        return {"nodes": nodes, "edges": edges}
