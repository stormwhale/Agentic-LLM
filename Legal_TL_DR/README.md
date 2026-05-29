# ⚖️ Legal/Personal Document Analyzer (TL:DR) (Work in Progress)

A Streamlit-based web application designed to store, manage, and extract key information from legal documents and websites. Built with privacy in mind, it uses LlamaIndex, ChromaDB, and local LLMs (via Ollama) to allow users to chat with their documents and get concise, TL;DR-style answers with exact source citations.

## ✨ Key Features

- **Document Ingestion**: Upload PDF documents locally or paste website URLs to extract and embed their content.
- **Smart Text Extraction**: Fast native PDF text extraction with an automatic fallback to GPU-accelerated OCR (EasyOCR) for scanned or image-based PDFs.
- **Vector Storage**: Uses ChromaDB for persistent local storage of document embeddings and metadata.
- **Interactive Chat**: Ask questions about specific documents or your entire database. The app uses LlamaIndex and a local LLM to provide accurate, TL;DR summaries.
- **Source Citations**: Every answer includes the retrieved evidence chunks ranked by relevance, showing you exactly where the information came from.
- **Library Browser**: Manage your knowledge base. View all embedded documents, filter by type (PDF/Web), search the database, preview AI-generated summaries, and delete documents.
- **Local & Private**: Powered entirely by local models via Ollama. No sensitive legal documents are sent to external APIs like OpenAI or Anthropic.

## 🛠️ Technology Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Backend/RAG**: [LlamaIndex](https://www.llamaindex.ai/)
- **Vector Database**: [ChromaDB](https://www.trychroma.com/)
- **LLM & Embeddings**: [Ollama](https://ollama.com/)
- **PDF & OCR**: PyMuPDF (`fitz`), EasyOCR, PIL, NumPy

## 🚀 Getting Started

### Prerequisites

1. **Python 3.8+** installed on your machine.
2. **Ollama** with **Qwen2.5:7B** and **nomic-embed-text:latest** installed and running locally. [Download Ollama here](https://ollama.com/).

### Installation

1. Clone the repository and navigate to the project directory:
   ```bash
   cd Legal_TL_DR
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Pull the required models using Ollama:
   ```bash
   # Main language model for chat and summarization
   ollama pull qwen2.5:7b
   
   # Embedding model for vectorizing documents
   ollama pull nomic-embed-text:latest
   ```

### Running the App

Start the Streamlit server:

```bash
streamlit run app.py
```

The app will open in your default web browser (typically at `http://localhost:8501`).

## 📂 Project Structure

- `app.py`: The main Streamlit frontend application containing the UI, Chat Box, and Library Browser.
- `backend.py`: The core logic handling document ingestion, OCR, ChromaDB interactions, LlamaIndex setup, and LLM querying.
- `chroma_db/`: Local directory where ChromaDB stores the vector embeddings and document registry.
- `requirements.txt`: List of Python dependencies.

## 💡 Usage Tips

- **Search Modes**: In the Chat Box, you can toggle between searching "Selected Documents" (great for cross-referencing specific cases) or the "Entire Database" (for broad knowledge retrieval).
- **Checking Sources**: Always check the "Retrieved Evidence" sidebar after asking a question to verify the LLM's claims against the original document text.
- **OCR Performance**: If you are uploading large scanned PDFs, the EasyOCR fallback may take some time. Having a compatible GPU will significantly speed up this process.
