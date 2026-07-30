import pytest
from src.config import load_config
from src.rag import retrieve_chunks

@pytest.fixture
def config():
    return load_config("config.yaml")

def test_config_loading(config):
    assert config.chunking.chunk_size > 0
    assert config.retrieval.top_k > 0

def test_empty_query_handling(config):
    # Pass dummy vector store
    result = retrieve_chunks("", vector_store=None, config=config)
    assert result == []