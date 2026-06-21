This directory contains research, scratchpads, and exploratory scripts used to build and validate the core data pipelines before migrating them to production modules in `src/`.

---

## 📔 Notebook Directory Registry

### `01_cfba_eda.ipynb`
* **Objective:** Audit the raw structural layout and quality of the CFPB database.
* **Key Operations:**
  * Calculates distribution ratios across different product groupings.
  * Measures narrative word length skewness to isolate outliers.
  * Discovers missing data patterns (e.g., matching entries with vs. without consumer written text).
  * Validates regex scripts designed to drop legal preambles and boilerplate introductory syntax.
  * Implements the structural group-merge strategy to clean up data mapping issues.

### `02_chunking_and_embedding.ipynb`
* **Objective:** Experiment with text partitioning strategies and vector generation settings.
* **Key Operations:**
  * Implements stratified random sampling logic to ensure subcategory weights match the raw metrics perfectly.
  * Tests text slicing behavior by tuning token/character boundaries (`chunk_size` and `chunk_overlap`).
  * Measures vector computation runtimes using `sentence-transformers`.
  * Generates, serializes, and runs localized query diagnostic checks against standard FAISS indices.

---

## 🛠️ Development Practices & Guidelines