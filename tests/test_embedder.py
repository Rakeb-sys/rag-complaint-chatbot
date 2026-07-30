import os
import shutil
import pytest
import pandas as pd
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer

# Import functions directly from src/embedder
from src.embedder import (
    chunk_dataframe,
    load_embedding_model,
    embed_chunks,
    build_chroma_store,
)

# Sample DataFrame fixture for testing
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Issue": ["Billing", "Debt collection"],
        "Consumer complaint narrative": [
            "I was charged twice for my credit card payment. This caused an overdraft fee.",
            "An agent called me multiple times a day regarding a debt I already paid off."
        ]
    })


def test_chunk_dataframe(sample_df):
    """Test 1: Verify dataframe chunking generates valid text and metadata chunks."""
    text_col = "Consumer complaint narrative"
    chunks = chunk_dataframe(sample_df, text_col=text_col)
    
    assert len(chunks) > 0, "Chunk list should not be empty."
    assert "text" in chunks[0], "Each chunk must contain a 'text' key."
    assert "metadata" in chunks[0], "Each chunk must contain a 'metadata' key."
    assert chunks[0]["metadata"]["Issue"] == "Billing"


def test_load_embedding_model():
    """Test 2: Verify embedding model loads successfully into SentenceTransformer."""
    model_name = "all-MiniLM-L6-v2"
    model = load_embedding_model(model_name)
    
    assert isinstance(model, SentenceTransformer), "Model should be a SentenceTransformer instance."


def test_embed_chunks():
    """Test 3: Verify embeddings match expected dimensions (384 for MiniLM-L6) and output structure."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    chunks = [
        {"text": "First test narrative sentence for embedding.", "metadata": {"id": 1}},
        {"text": "Second test narrative sentence for embedding.", "metadata": {"id": 2}}
    ]
    
    embeddings = embed_chunks(chunks, model)
    
    assert isinstance(embeddings, np.ndarray), "Embeddings should be returned as a NumPy array."
    assert len(embeddings) == 2, "Should return 2 embedding vectors."
    assert embeddings.shape[1] == 384, "MiniLM-L6 output dimension should be 384."


def test_build_chroma_store(tmp_path):
    """Test 4: Verify ChromaDB store persists documents and creates expected collection."""
    test_persist_dir = str(tmp_path / "test_chroma")
    chunks = [
        {"text": "Sample text for ChromaDB vector index.", "metadata": {"category": "test"}}
    ]
    embeddings = np.random.rand(1, 384).astype(np.float32)

    collection = build_chroma_store(chunks, embeddings, persist_dir=test_persist_dir)
    
    assert collection is not None
    assert collection.count() == 1, "ChromaDB collection should contain 1 record."


def test_end_to_end_embedding_pipeline(sample_df, tmp_path):
    """Test 5: Integration test running chunking -> embedding -> Chroma storing."""
    test_persist_dir = str(tmp_path / "test_chroma_pipeline")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # 1. Chunk
    chunks = chunk_dataframe(sample_df, text_col="Consumer complaint narrative")
    # 2. Embed
    embeddings = embed_chunks(chunks, model)
    # 3. Store
    collection = build_chroma_store(chunks, embeddings, persist_dir=test_persist_dir)
    
    assert collection.count() == len(chunks), "All created chunks should be stored in ChromaDB."