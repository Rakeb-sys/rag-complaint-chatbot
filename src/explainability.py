import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def create_shap_explainer(model, X_train: pd.DataFrame):
    """Initializes the appropriate SHAP explainer based on model type."""
    try:
        # Optimized tree explainer for XGBoost / LightGBM / Random Forest
        explainer = shap.TreeExplainer(model)
        logger.info("Initialized SHAP TreeExplainer.")
    except Exception:
        # Fallback KernelExplainer or generic Explainer
        explainer = shap.Explainer(model, X_train)
        logger.info("Initialized generic SHAP Explainer.")
    
    return explainer


def compute_shap_values(explainer, X_data: pd.DataFrame):
    """Computes SHAP values for given dataset features."""
    shap_values = explainer(X_data)
    return shap_values


def plot_global_summary(shap_values, max_display: int = 10, save_path: str = None):
    """Q1: Global Feature Importance (Beeswarm plot).
    
    Shows feature importance ranking and directional impact on prediction outputs.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_local_waterfall(shap_values, sample_index: int = 0, save_path: str = None):
    """Q2: Local Single-Prediction Explanation (Waterfall plot).
    
    Explains how baseline prediction was pushed up/down by individual feature values.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.plots.waterfall(shap_values[sample_index], show=False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_dependence_pattern(shap_values, feature_name: str, X_data: pd.DataFrame, save_path: str = None):
    """Q3: Identifying Concerning Patterns (Dependence plot).
    
    Shows how the effect of a single feature varies across its range and interacts with others.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.plots.scatter(shap_values[:, feature_name], color=shap_values, show=False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig