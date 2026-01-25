"""
USAGE
  1. Type-safe environment variable loading
  2. Automatic validation (wrong types fail fast)
  3. Default values with override capability
  4. Documentation via type hints
  5. Nested configuration groups

LANGCHAIN RELEVANCE:

SAMPLE CODE
---------------
# Import package
from src.config.settings import settings

# Use in code
llm = ChatGoogleGenerativeAI(
    model=settings.llm.model,
    temperature=settings.llm.temperature,
    google_api_key=settings.google_api_key,
)
"""

from functools import lru_cache 
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class LLMSettings(BaseSettings):
    """
    class for all configuration for LLM project

    """
    model_config = SettingsConfigDict(env_prefix='GEMINI_')

    #model name
    model:str = Field(
        default="gemini-2.5-flash",
        description="Gemini model name for text generation"
    )

    temperature:float = Field(
        default=0.2,
        ge=0.0,
        le = 1.0,
        description= "Generation temperature"
    )
    #Max token used for respond
    max_tokens: int = Field (
        default= 2048,
        alias = "max_output_tokens",
        description="Maximum output tokens"
    )


class EmbeddingSettings(BaseSettings):
    """
    Configuration for the embedding model.
    
   Separate from LLM setting as:
      - LLM: Text generation (expensive, smart)
      - Embedding: Text → Vector (cheap, fast)
    """
    model_config = SettingsConfigDict(env_prefix="EMBEDDING_")

    model:str = Field(
        default="models/text-embedding-004",
        description="Embedding model")
     # Batch size for embedding multiple texts
    batch_size:int = Field(
        default=100,
        ge=1,
        le=500,
        description="Batch size for embedding request"
    )

class ChromaSettings(BaseSettings):
    """
    Configuration for Chroma vector DB
    """
    model_config = SettingsConfigDict(env_prefix="CHROMA_")

    persist_dir:Path = Field(
        default=Path("./data/chromadb"),
        description="Directory for ChromaDB persistence"
    )

    collection:str = Field(
        default="jira_issues",
        alias = "collection_name",
        description="Collection name for Jira tickets"
    )

class RAGSettings(BaseSettings):
    """
    Configuration for Retrieval-Augmented Generation.
    
    Main Parameters:
    - top_k: More = richer context but higher latency/cost
    - chunk_size: Larger = more context per chunk, fewer chunks
    - chunk_overlap: More = better continuity, more redundancy
    """
    model_config = SettingsConfigDict(env_prefix="RAG_")

    top_k:int = Field(
        default=6,
        ge=1,
        le=20,
        description="Number of documents to retrieve"
    )
    chunk_size:int = Field(
        default=1200,
        ge=100,
        le=4000,
        description="Text chunk size in characters"
    )
    chunk_overlap:int = Field(
        default=200,
        ge=0,
        le=1000,
        description="Overlap between chunks"
    )

    # @field_validator("chunk_overlap")
    # @classmethod
    # def validate_overlap(cls, v: int, info) -> int:
    #     """Ensure overlap is less than chunk size."""
    #     chunk_size = info.data.get()
    #     return v

class ServerSettings(BaseSettings):
    """
    Configuration for the FastAPI server.
    """
    model_config = SettingsConfigDict(env_prefix="")
    port:int = Field(
        default=8000,
        ge = 1,
        le=65535,
        description="Server port"
    )
    env: str = Field(
        default="development",
        description="Environment (development/staging/production)"
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    
    cors_origins: str = Field(
        default="http://localhost:3000,http://localhost:80",
        description="Comma-separated CORS origins"
    )
    
    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins into a list."""
        return [origin.strip() for origin in self.cors_origins.split(",")]
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.env.lower() == "development"
    

#Main setting class
#Aggreate all configration group + top level setting

class Settings(BaseSettings):
    """
    ARCHITECTURE PATTERN: Singleton via lru_cache
    We use @lru_cache on get_settings() to ensure only one Settings
    instance exists. This is important because:
      1. Environment is read once, not repeatedly
      2. Validation happens once at startup
      3. All code shares the same config
    
    USAGE:
        from src.config.settings import settings
        print(settings.google_api_key)
        print(settings.llm.model)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra = 'ignore',
        case_sensitive=False
    )

    #API Key
    google_api_key: str = Field(
        default="",
        description="Google API key for Gemini"
    )

    #Nested Configuration Group
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    chroma: ChromaSettings = Field(default_factory=ChromaSettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)

    #Compute properties
    @property
    def project_root(self) -> Path:
        return Path(__file__).resolves().parents[2]
    
    @property
    def data_dir(self) -> Path:
        """Get Data directory"""
        return self.project_root/"data"
    @field_validator("google_api_key")
    @classmethod
    def validate_api_key(cls,v:str)-> str:
        """Warning if API missing"""
        if not v or v =='google-api-key-here':#placeholder for putting in api :
             import warnings
             warnings.warn(
                 "GOOGLE_API_KEY is not set. API calls will fail. "
                "Set it in .env or as an environment variable."
             )
        return v
    
# SECTION 3: SINGLETON ACCESS
@lru_cache
def get_settings() -> Settings:
    """
    Get the application settings (singleton).
    
    Uses lru_cache to ensure only one instance is created.
    This is the recommended pattern for Pydantic Settings.
    
    Returns:
        Settings: Application configuration
    """
    return Settings()

settings = get_settings()

