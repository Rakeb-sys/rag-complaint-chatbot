import os
import joblib
import pandas as pd
import pytest
from xgboost import XGBClassifier
from src.train_model import train_and_save


@pytest.fixture
def mock_processed_csv(tmp_path):
    """Creates dummy feature data resembling processed complaints."""
    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "filtered_complaints.csv"

    df_mock = pd.DataFrame({
        "complaint_id": [1, 2, 3, 4, 5, 6],
        "narrative_length": [150, 450, 200, 800, 300, 1200],
        "word_count": [25, 80, 35, 140, 50, 210],
        "days_to_response": [1, 5, 2, 10, 3, 15],
        "target": [0, 1, 0, 1, 0, 1],
    })
    df_mock.to_csv(csv_path, index=False)
    return csv_path


def test_train_and_save_pipeline(mock_processed_csv, monkeypatch, tmp_path):
    # Redirect output model folder to temporary path
    model_dir = tmp_path / "models"
    model_path = model_dir / "xgb_model.pkl"

    # Execute training routine
    train_and_save(
        data_path=str(mock_processed_csv),
        model_output_path=str(model_path)
    )

    # Assertions
    assert os.path.exists(model_path), "Model artifact xgb_model.pkl was not saved."
    
    loaded_model = joblib.load(model_path)
    assert isinstance(loaded_model, XGBClassifier), "Saved artifact is not an XGBClassifier."
    
    # Verify expected feature names match dataset
    if hasattr(loaded_model, "feature_names_in_"):
        assert "narrative_length" in loaded_model.feature_names_in_
        assert "target" not in loaded_model.feature_names_in_
        assert "complaint_id" not in loaded_model.feature_names_in_