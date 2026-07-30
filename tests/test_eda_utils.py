import os
import pytest
import pandas as pd
from src.eda_utils import (
    set_aesthetics,
    load_data,
    assess_quality,
    clean_text_noise,
    normalize_text,
)


def test_clean_text_noise_urls_and_html():
    raw_text = "Check out <a href='http://example.com'>this link</a> for details."
    cleaned = clean_text_noise(raw_text)
    assert "http" not in cleaned
    assert "example.com" not in cleaned
    assert "<a>" not in cleaned
    assert "<a" not in cleaned


def test_clean_text_noise_phone_and_punctuation():
    raw_text = "Call me at 800-555-0199 or 555-0199! Thanks."
    cleaned = clean_text_noise(raw_text)
    assert "800-555-0199" not in cleaned
    assert "555-0199" not in cleaned
    assert "!" not in cleaned
    # Check that keywords remain and numbers/punctuation were stripped
    assert "call me at" in cleaned
    assert "thanks" in cleaned


def test_normalize_text():
    # Tests stopword removal and verb/noun lemmatization ("running" -> "run", "complaints" -> "complaint")
    raw_text = "the customer was running and submitting multiple complaints"
    cleaned = clean_text_noise(raw_text)
    normalized = normalize_text(cleaned)
    
    assert "the" not in normalized.split()  # Stopword removed
    assert "was" not in normalized.split()  # Stopword removed
    assert "run" in normalized.split()      # Verb lemmatized
    assert "complaint" in normalized.split()# Noun lemmatized


def test_load_data(tmp_path):
    # Create a temporary CSV file for testing
    df_stub = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    file_path = tmp_path / "sample.csv"
    df_stub.to_csv(file_path, index=False)

    df_loaded = load_data(str(file_path))
    assert df_loaded.shape == (2, 2)
    assert list(df_loaded.columns) == ["col1", "col2"]


def test_assess_quality(capsys):
    # Test DataFrame with missing values
    df_missing = pd.DataFrame({"a": [1, None, 3], "b": [None, None, "x"]})
    assess_quality(df_missing)
    captured = capsys.readouterr()
    assert "Columns with missing values:" in captured.out

    # Test DataFrame without missing values
    df_clean = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    assess_quality(df_clean)
    captured_clean = capsys.readouterr()
    assert "No missing values detected." in captured_clean.out


def test_set_aesthetics():
    # Ensure set_aesthetics runs cleanly without throwing errors
    try:
        set_aesthetics()
        assert True
    except Exception as e:
        pytest.fail(f"set_aesthetics raised an exception: {e}")