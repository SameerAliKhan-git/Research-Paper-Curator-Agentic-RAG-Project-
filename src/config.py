import os
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).parent.parent
ENV_FILE_PATH = PROJECT_ROOT / ".env"


class BaseConfigSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        extra="ignore",
        frozen=True,
        env_nested_delimiter="__",
        case_sensitive=False,
    )


class ArxivSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="ARXIV__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    base_url: str = "https://export.arxiv.org/api/query"
    pdf_cache_dir: str = "./data/arxiv_pdfs"
    rate_limit_delay: float = 3.0
    timeout_seconds: int = 30
    max_results: int = 15
    search_category: str = "cs.AI"
    download_max_retries: int = 3
    download_retry_delay_base: float = 5.0
    max_concurrent_downloads: int = 5
    max_concurrent_parsing: int = 1

    namespaces: dict = {
        "atom": "http://www.w3.org/2005/Atom",
        "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
        "arxiv": "http://arxiv.org/schemas/atom",
    }

    @field_validator("pdf_cache_dir")
    @classmethod
    def validate_cache_dir(cls, v: str) -> str:
        os.makedirs(v, exist_ok=True)
        return v


class PDFParserSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="PDF_PARSER__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    max_pages: int = 30
    max_file_size_mb: int = 20
    do_ocr: bool = False
    do_table_structure: bool = True


class ChunkingSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="CHUNKING__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    chunk_size: int = 600  # Target words per chunk
    overlap_size: int = 100  # Words to overlap between chunks
    min_chunk_size: int = 100  # Minimum words for a valid chunk
    section_based: bool = True  # Use section-based chunking when available


class OpenSearchSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="OPENSEARCH__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    host: str = "http://localhost:9200"
    index_name: str = "arxiv-papers"
    chunk_index_suffix: str = "chunks"  # Creates single hybrid index: {index_name}-{suffix}
    max_text_size: int = Field(default=1000000, ge=1000, le=10000000)
    request_timeout: int = Field(default=30, ge=5, le=120, description="OpenSearch request timeout in seconds")

    # Vector search settings
    vector_dimension: int = Field(default=1024, ge=128, le=4096)  # Jina embeddings dimension
    vector_space_type: str = "cosinesimil"  # cosinesimil, l2, innerproduct

    # Hybrid search settings
    rrf_pipeline_name: str = "hybrid-rrf-pipeline"
    hybrid_search_size_multiplier: int = Field(default=2, ge=1, le=5)  # Get k*multiplier for better recall


class LangfuseSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="LANGFUSE__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    public_key: str = ""
    secret_key: str = ""
    host: str = "http://localhost:3000"  # Self-hosted Langfuse URL
    enabled: bool = True
    flush_at: int = 15  # Number of events before flushing
    flush_interval: float = 1.0  # Seconds between flushes
    max_retries: int = 3
    timeout: int = 30
    debug: bool = False


class RedisSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="REDIS__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    host: str = "localhost"
    port: int = Field(default=6379, ge=1, le=65535)
    password: str = ""
    db: int = Field(default=0, ge=0, le=15)
    decode_responses: bool = True
    socket_timeout: int = Field(default=30, ge=5, le=120)
    socket_connect_timeout: int = Field(default=30, ge=5, le=120)

    # Cache settings
    ttl_hours: int = Field(default=6, ge=1, le=168)  # Cache TTL in hours (1 week max)


class TelegramSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="TELEGRAM__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    bot_token: str = ""
    enabled: bool = False


class RerankerSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="RERANKER__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    enabled: bool = True  # Enable by default; gracefully degrades if dependencies missing
    provider: str = "bge"  # jina, cohere, bge (local)
    model: str = "BAAI/bge-reranker-v2-m3"
    api_key: str = ""
    base_url: str = "https://api.jina.ai/v1/rerank"
    timeout: float = 30.0
    top_n: int = 3


class JWTSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="JWT__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    secret_key: str = "your-super-secret-jwt-key-change-it-in-production"
    algorithm: str = "HS256"
    expire_minutes: int = 1440  # 1 day expiration


class EmailSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="EMAIL__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    from_address: str = "noreply@arxivcurator.com"
    enabled: bool = False


class Settings(BaseConfigSettings):
    app_version: str = "0.1.0"
    debug: bool = False
    environment: Literal["development", "staging", "production"] = "development"
    service_name: str = "rag-api"

    postgres_database_url: str = ""
    postgres_echo_sql: bool = False
    postgres_pool_size: int = Field(default=20, ge=1, le=100)
    postgres_max_overflow: int = Field(default=0, ge=0, le=100)

    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "gemma3:latest"
    ollama_timeout: int = Field(default=300, ge=30, le=600)

    # Jina AI embeddings configuration
    jina_api_key: str = ""

    # Google Gemini cloud LLM configuration
    gemini_api_key: str = ""
    llm_provider: str = "ollama"  # ollama or gemini

    # LLM resource limits
    llm_max_tokens: int = Field(default=2048, ge=128, le=8192, description="Max tokens for LLM generation")
    llm_max_concurrent: int = Field(default=10, ge=1, le=100, description="Max concurrent LLM requests")

    # Rate limiting defaults
    default_rate_limit: int = Field(default=60, ge=1, le=1000, description="Default requests per minute per API key")

    # Request size limit
    max_request_size_mb: int = Field(default=10, ge=1, le=100, description="Max request body size in MB")

    # Feature flags for pipeline components
    enable_semantic_cache: bool = Field(default=True, description="Enable semantic cache for RAG")
    enable_exact_cache: bool = Field(default=True, description="Enable exact match cache for RAG")
    enable_reranker: bool = Field(default=True, description="Enable reranker in search pipeline")
    enable_hybrid_search: bool = Field(default=True, description="Enable hybrid BM25+vector search")
    enable_langfuse_tracing: bool = Field(default=True, description="Enable Langfuse observability tracing")
    enable_multimodal: bool = Field(default=False, description="Enable multimodal processing (figures extraction & indexing)")
    enable_jwt_auth: bool = Field(default=True, description="Enable JWT-based user authentication")
    enable_parent_child: bool = Field(default=True, description="Enable Parent-Child chunking strategy")

    arxiv: ArxivSettings = Field(default_factory=ArxivSettings)
    pdf_parser: PDFParserSettings = Field(default_factory=PDFParserSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    opensearch: OpenSearchSettings = Field(default_factory=OpenSearchSettings)
    langfuse: LangfuseSettings = Field(default_factory=LangfuseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    telegram: TelegramSettings = Field(default_factory=TelegramSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    jwt: JWTSettings = Field(default_factory=JWTSettings)
    email: EmailSettings = Field(default_factory=EmailSettings)

    @field_validator("postgres_database_url")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        if not v:
            return v
        if not (v.startswith("postgresql://") or v.startswith("postgresql+psycopg2://")):
            raise ValueError("Database URL must start with 'postgresql://' or 'postgresql+psycopg2://'")
        return v

    @field_validator("jina_api_key")
    @classmethod
    def validate_jina_key(cls, v: str) -> str:
        if v and v != "your_jina_api_key_here" and not v.startswith("jina_"):
            raise ValueError("Jina API key should start with 'jina_'")
        return v

    def log_safe_summary(self) -> dict:
        """Return a safe summary of settings for logging (no secrets)."""
        return {
            "environment": self.environment,
            "service_name": self.service_name,
            "ollama_host": self.ollama_host,
            "ollama_model": self.ollama_model,
            "opensearch_host": self.opensearch.host,
            "opensearch_index": self.opensearch.index_name,
            "redis_host": self.redis.host,
            "redis_port": self.redis.port,
            "jina_api_key_set": bool(self.jina_api_key and self.jina_api_key != "your_jina_api_key_here"),
            "langfuse_enabled": self.langfuse.enabled,
            "telegram_enabled": self.telegram.enabled,
            "enable_semantic_cache": self.enable_semantic_cache,
            "enable_reranker": self.enable_reranker,
        }


_settings_cache: Settings | None = None


def get_settings() -> Settings:
    """Return a cached Settings singleton (Settings is frozen, safe to share)."""
    global _settings_cache
    if _settings_cache is None:
        _settings_cache = Settings()
    return _settings_cache
