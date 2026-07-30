import os
import joblib
import pandas as pd
import gradio as gr
import shap
import matplotlib.pyplot as plt

from src.rag_pipeline import RAGPipeline
from src.config import cfg
from src.explainability import (
    create_shap_explainer,
    compute_shap_values,
    plot_global_summary,
    plot_local_waterfall,
    plot_dependence_pattern,
)

# --- Configuration & Initialization ---
CHROMA_DIR = os.getenv("CHROMA_DIR", "vector_store/chroma")
LLM_MODEL = cfg.models.llm_model

rag = RAGPipeline(
    chroma_dir=CHROMA_DIR,
    llm_model=LLM_MODEL,
    k=3,
    max_new_tokens=100,
)

PRODUCT_CHOICES = [
    "All Products",
    "Credit Card",
    "Personal Loan",
    "Savings Account",
    "Money Transfer",
]

# --- Helper Functions for RAG ---
def format_sources(sources: list[dict]) -> str:
    if not sources:
        return "No sources retrieved."
    parts = []
    for i, s in enumerate(sources, 1):
        meta = s.get("metadata", {})
        product = meta.get("product_category", "N/A")
        issue = meta.get("issue", "N/A")
        company = meta.get("company", "N/A")
        date = meta.get("date_received", "N/A")
        similarity = round(1 - float(s.get("distance", 0)), 3)
        parts.append(
            f"**Source {i}** | Product: {product} | Issue: {issue} | "
            f"Company: {company} | Date: {date} | Similarity: {similarity}\n\n"
            f"> {s['text'][:300]}{'…' if len(s['text']) > 300 else ''}"
        )
    return "\n\n---\n\n".join(parts)


def answer_question(question: str, product_filter: str, history: list):
    if not question.strip():
        return history, "Sources will appear here.", ""

    product = None if product_filter == "All Products" else product_filter

    try:
        result = rag.run(question, product_filter=product)
        answer = result["answer"]
        sources_md = format_sources(result["sources"])
    except Exception as e:
        answer = f"Error generating answer: {e}"
        sources_md = ""

    history = history or []
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    return history, sources_md, ""


# --- Model & SHAP Data Loading ---
def load_explainer_and_data():
    data_path = "data/processed/filtered_complaints.csv"
    model_path = "models/xgb_model.pkl"

    if not os.path.exists(data_path) or not os.path.exists(model_path):
        return pd.DataFrame(), None, None

    df = pd.read_csv(data_path)
    model = joblib.load(model_path)

    # 1. Compute engineered features if missing from CSV
    if "narrative_length" not in df.columns:
        # Find the text narrative column (or use a default empty/str column)
        text_cols = [c for c in df.columns if "narrative" in c.lower() or "text" in c.lower()]
        if text_cols:
            df["narrative_length"] = df[text_cols[0]].astype(str).str.len()
        else:
            df["narrative_length"] = 0

    # 2. Retrieve expected features from the XGBoost model safely
    if hasattr(model, "feature_names_in_"):
        expected_features = list(model.feature_names_in_)
    elif hasattr(model, "get_booster") and model.get_booster().feature_names:
        expected_features = model.get_booster().feature_names
    else:
        expected_features = list(df.select_dtypes(include=["number"]).columns)

    # 3. Filter DataFrame using only available/computed expected features
    missing_cols = [col for col in expected_features if col not in df.columns]
    for col in missing_cols:
        df[col] = 0  # Fill missing expected features with 0 as fallback

    X = df[expected_features]

    # 4. Generate SHAP values
    explainer = create_shap_explainer(model, X)
    shap_vals = compute_shap_values(explainer, X)

    return X, explainer, shap_vals

X_data, explainer_obj, shap_values_obj = load_explainer_and_data()


# --- SHAP Plot Generator Functions ---
def get_global_plot():
    if shap_values_obj is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "SHAP data or model not loaded.", ha="center")
        return fig
    return plot_global_summary(shap_values_obj)


def get_local_plot(sample_idx):
    if shap_values_obj is None or X_data.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "SHAP data or model not loaded.", ha="center")
        return fig
    idx = int(sample_idx) if 0 <= sample_idx < len(X_data) else 0
    return plot_local_waterfall(shap_values_obj, sample_index=idx)


def get_dependence_plot(feature_name):
    if shap_values_obj is None or not feature_name:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "SHAP data or model not loaded.", ha="center")
        return fig
    return plot_dependence_pattern(shap_values_obj, feature_name, X_data)


# --- Gradio UI Layout ---
with gr.Blocks(title="CrediTrust Complaint Analyst", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# CrediTrust Financial — Complaint Analyst & Model Insights")

    with gr.Tabs():
        # --- TAB 1: RAG Chatbot ---
        with gr.TabItem("💬 Complaint Analyst Chatbot"):
            gr.Markdown(
                """
                Ask questions about customer complaints across Credit Cards, Personal Loans,
                Savings Accounts, and Money Transfers. Answers are grounded in real CFPB complaint data.
                """
            )
            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(label="Conversation", height=450)
                    with gr.Row():
                        question_box = gr.Textbox(
                            placeholder="e.g. Why are customers unhappy with credit cards?",
                            label="Your Question",
                            lines=2,
                            scale=4,
                        )
                        product_dropdown = gr.Dropdown(
                            choices=PRODUCT_CHOICES,
                            value="All Products",
                            label="Filter by Product",
                            scale=1,
                        )
                    with gr.Row():
                        submit_btn = gr.Button("Ask", variant="primary")
                        clear_btn = gr.Button("Clear")

                with gr.Column(scale=1):
                    sources_box = gr.Markdown(value="Sources will appear here.")

            state_history = gr.State([])

            submit_btn.click(
                fn=answer_question,
                inputs=[question_box, product_dropdown, state_history],
                outputs=[chatbot, sources_box, question_box],
            )
            question_box.submit(
                fn=answer_question,
                inputs=[question_box, product_dropdown, state_history],
                outputs=[chatbot, sources_box, question_box],
            )
            clear_btn.click(
                fn=lambda: ([], [], "Sources will appear here.", ""),
                outputs=[chatbot, state_history, sources_box, question_box],
            )

            gr.Examples(
                examples=[
                    ["Why are people unhappy with Credit Cards?", "Credit Card"],
                    ["What are the most common loan repayment complaints?", "Personal Loan"],
                    ["Are there fraud complaints in savings accounts?", "Savings Account"],
                    ["What issues do customers face with money transfers?", "Money Transfer"],
                    ["Which product has the most unresolved complaints?", "All Products"],
                ],
                inputs=[question_box, product_dropdown],
            )

        # --- TAB 2: Model Explainability (SHAP) ---
        with gr.TabItem("📊 Model Explainability (SHAP)"):
            gr.Markdown("## ML Model Prediction Diagnostics")
            
            with gr.Tabs():
                with gr.TabItem("Global Importance"):
                    gr.Markdown("Top features influencing model predictions across the entire dataset.")
                    global_plot_btn = gr.Button("Generate Global Summary")
                    global_plot_output = gr.Plot()
                    global_plot_btn.click(fn=get_global_plot, outputs=global_plot_output)

                with gr.TabItem("Local Prediction Explainer"):
                    gr.Markdown("Explain individual prediction outcomes using waterfall charts.")
                    sample_number = gr.Number(label="Select Complaint Sample Index", value=0, precision=0)
                    local_plot_btn = gr.Button("Explain Prediction")
                    local_plot_output = gr.Plot()
                    local_plot_btn.click(fn=get_local_plot, inputs=sample_number, outputs=local_plot_output)

                with gr.TabItem("Dependence & Pattern Analysis"):
                    gr.Markdown("Examine interactions and non-linear feature relationships.")
                    feature_selector = gr.Dropdown(
                        choices=list(X_data.columns) if not X_data.empty else [],
                        label="Select Feature",
                        value=X_data.columns[0] if not X_data.empty else None
                    )
                    dep_plot_btn = gr.Button("Plot Dependence")
                    dep_plot_output = gr.Plot()
                    dep_plot_btn.click(fn=get_dependence_plot, inputs=feature_selector, outputs=dep_plot_output)

if __name__ == "__main__":
    demo.launch(share=False, server_port=7860)