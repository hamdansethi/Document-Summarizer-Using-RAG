# Document Summarization with Google Gemini + LangChain

This project contains two summarization scripts:

- `simple_summarizer.py`: a basic summarization pipeline.
- `advanced_summarizer.py`: a more powerful, scalable, and robust version with vector search and API key rotation.

---

## What `simple_summarizer.py` Does

This script is a lightweight, single-pass summarizer designed for quick use cases.

### Features:
- Loads a `.pdf` or `.txt` file.
- Breaks content into paragraphs or simple chunks.
- Uses Gemini's `gemini-pro` model to generate a summary.
- Good for short documents or quick drafts.

### Limitations:
- No semantic chunking — chunks may be too small or too long.
- No vector store or context retrieval — can miss important context.
- Only supports a single API key.
- Fails on large documents due to prompt length limits.

---

## What `advanced_summarizer.py` Adds

This is a more advanced, production-ready summarization pipeline.

### Key Features:

#### Semantic Chunking
- Uses **LangChain's `SemanticChunker`** to break content into meaningful units.
- Based on **embedding similarity**, not just paragraph breaks.

#### Hierarchical Summarization
- First summarizes individual chunks.
- Then creates a **final summary from those summaries**, preserving key ideas better.

#### Vector Store with ChromaDB
- All chunks are embedded and stored in **Chroma vector database**.
- Enables context retrieval using **MMR (Maximal Marginal Relevance)**.

#### Multi-API Key Support
- Loads a list of **Google API keys from `.env`** file.
- Automatically rotates API keys on rate limit or quota errors.

#### Retry Logic
- On failure, retries with the next available API key.
- Prevents the script from crashing when one key is exhausted.

#### Performance Logging
- Measures and prints **latency (runtime)**.
- Displays **similarity scores** from the retrieved contexts.

---

## Technologies Used

- **LangChain** – for pipeline orchestration and chunking
- **Google Gemini (`gemini-pro`, `gemini-embedding-001`)** – LLM + embeddings
- **ChromaDB** – vector store for semantic search
- **dotenv** – for managing API keys securely
- **Python** – of course :)

---

## Setup Instructions

1. Clone this repo.

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root:

   ```env
   GOOGLE_API_KEYS=your_key_1,your_key_2,your_key_3
   ```

4. Run either script:

   ```bash
   python simple_summarizer.py
   ```

   or

   ```bash
   python advanced_summarizer.py
   ```

---

## Example Output (Advanced Summarizer)

```text
Document split into 22 semantic chunks.
Running hierarchical summarization...

=== Document Summary ===

[Concise structured summary here...]

=== Retrieved Contexts ===

--- Context 1 ---
[Most relevant semantic chunk]

--- Context 2 ---
[...]

Latency: 12.45 seconds
Similarity Scores: [0.89, 0.87, 0.85, ...]
```

---

## Recommendation

If you're just testing or summarizing small documents, start with `simple_summarizer.py`.

But if you want robust summarization with better context understanding, automatic retry, and scale support — go with `advanced_summarizer.py`.

---

## Questions?

Feel free to open an issue or reach out with questions or improvements!

```

---