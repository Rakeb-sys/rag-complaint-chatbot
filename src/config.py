import yaml
from pathlib import Path
from pydantic import BaseModel

class PathConfig(BaseModel):
    raw_data: Path
    processed_data: Path
    vector_store: Path
    logs: Path

class ChunkConfig(BaseModel):
    chunk_size: int
    chunk_overlap: int

class ModelConfig(BaseModel):
    embedding_model: str
    llm_model: str

class RetrievalConfig(BaseModel):
    top_k: int
    similarity_threshold: float

class AppConfig(BaseModel):
    paths: PathConfig
    chunking: ChunkConfig
    models: ModelConfig
    retrieval: RetrievalConfig

def load_config(config_path: str = "config.yaml") -> AppConfig:
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)
    return AppConfig(**config_data)

# Singleton configuration instance
cfg = load_config()