import os
import gradio as gr
from src.rag_pipeline import RAGPipeline

CHROMA_DIR = os.getenv("CHROMA_DIR", "vector_store/chroma")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

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


def format_sources(sources: list[dict]) -> str:
    if not sources:
        return "No sources retrieved."
    parts = []
    for i, s in enumerate(sources, 1):
        meta = s["metadata"]
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


with gr.Blocks(title="CrediTrust Complaint Analyst", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # CrediTrust Financial — Complaint Analyst
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
            sources_box = gr.Markdown(label="Retrieved Sources", value="Sources will appear here.")

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


if __name__ == "__main__":
    demo.launch(share=False, server_port=7860)
