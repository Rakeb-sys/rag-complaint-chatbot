import pytest
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from src.explainability import create_shap_explainer, compute_shap_values


@pytest.fixture
def dummy_model_and_data():
    X = pd.DataFrame({
        "feature_a": np.random.rand(50),
        "feature_b": np.random.rand(50),
    })
    y = np.random.randint(0, 2, size=50)
    
    model = XGBClassifier(n_estimators=10, max_depth=2)
    model.fit(X, y)
    return model, X


def test_shap_pipeline(dummy_model_and_data):
    model, X = dummy_model_and_data
    
    explainer = create_shap_explainer(model, X)
    shap_vals = compute_shap_values(explainer, X)
    
    assert shap_vals is not None
    assert shap_vals.values.shape == X.shape, "SHAP values matrix must match input feature dimensions."