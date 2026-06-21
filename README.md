# rag-complaint-chatbot
Intelligent Complaint Analysis for Financial Services: Building a RAG-Powered Chatbot to Turn Customer Feedback into Actionable Insights

Dataset Source: 
https://drive.google.com/file/d/1zKOyxGlSlwSMPtisPJYUgXUevGn_KWO3/view?usp=drive_link
https://drive.google.com/file/d/1T-Rfs13riuawtf6MW0nhVuQmBQLKV9Fh/view?usp=sharing


# CrediTrust Financial: Internal RAG Complaint Chatbot

An enterprise-grade Retrieval-Augmented Generation (RAG) pipeline designed to process unstructured consumer complaint data from the Consumer Financial Protection Bureau (CFPB). This system transforms millions of dense text records into a semantic, searchable vector store, allowing non-technical stakeholders (Product Managers, Compliance Officers, and Support Leads) to extract evidence-backed operational insights via a natural language chat interface.

---

## 🛠️ System Architecture

The pipeline consists of four modular layers:
1. **Data Engineering:** Ingestion, cleaning, noise-reduction, and category consolidation of raw CFPB entries.
2. **Indexing Pipeline:** Strategic stratified sampling, text chunking via recursive character splitting, and dense vector generation.
3. **RAG Core Logic:** A localized vector retrieval node ($k=5$) integrated with an anchored system prompt and an LLM generation layer.
4. **Interface:** An interactive web application built with a side-by-side source verification dashboard.