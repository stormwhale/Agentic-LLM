# Streamlit: Step-by-Step Guide
> Tailored to your Legal TL;DR project

---

## How Streamlit Works

Streamlit re-runs your **entire `app.py` from top to bottom** every time:
- A user interacts with a widget
- You save the file (hot reload)
- `st.rerun()` is called

This means **order matters** and **state doesn't persist between reruns** unless you use `st.session_state`.

---

## Step 1 — Displaying Content

```python
import streamlit as st

# Page config (must be FIRST st call)
st.set_page_config(page_title="Legal TL;DR", page_icon="⚖️", layout="wide")

st.title("⚖️ Legal Document Analyzer")      # Big heading (H1)
st.header("Section Header")                  # H2
st.subheader("Subsection")                   # H3
st.write("Hello, world!")                    # Renders text, dicts, dataframes...
st.markdown("**Bold** and *italic* text")    # Full markdown support
st.code("print('hello')", language="python") # Syntax highlighted code
st.info("ℹ️ This is an info box")
st.success("✅ Document loaded!")
st.warning("⚠️ This may take a while...")
st.error("❌ Something went wrong.")
```

> [!TIP]
> `st.write()` is the Swiss-army knife — it auto-detects and renders strings, dicts, DataFrames, Matplotlib figures, and more.

---

## Step 2 — Layout (Sidebar, Columns, Expander)

```python
# Sidebar — great for controls/settings
with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Upload PDF", type="PDF")

# Columns — split the main area horizontally
col1, col2 = st.columns(2)
with col1:
    st.write("Left panel")
with col2:
    st.write("Right panel")

# Ratio columns (3:1 split)
col_main, col_side = st.columns([3, 1])

# Expander — collapsible section
with st.expander("Show source citations"):
    st.write("Citation text here...")

# Tabs
tab1, tab2 = st.tabs(["Summary", "Raw Text"])
with tab1:
    st.write("Summary goes here")
with tab2:
    st.write("Raw text goes here")
```

---

## Step 3 — Widgets (User Input)

```python
# Text input
query = st.text_input("Ask a question about the document:")
query_long = st.text_area("Or type a longer query:", height=100)

# File uploader (you're already using this!)
pdf = st.file_uploader("Upload PDF", type=["pdf"])

# URL input
url = st.text_input("Or enter a URL:")

# Button
if st.button("Analyze Document"):
    st.write("Analyzing...")

# Select box / Radio
mode = st.selectbox("Input mode", ["PDF Upload", "URL"])
mode2 = st.radio("Choose mode", ["PDF", "URL"])

# Slider
top_k = st.slider("Number of sources to retrieve", min_value=1, max_value=10, value=5)

# Checkbox
show_debug = st.checkbox("Show debug info")
```

> [!IMPORTANT]
> Every widget returns its current value. Capture it in a variable: `query = st.text_input(...)`. The app re-runs when the user changes the widget, giving you the new value.

---

## Step 4 — Session State (Persisting Data Between Reruns)

This is the **most important concept**. Without it, your index/filters are lost on every rerun.

```python
# Initialize state keys (do this at the top of app.py)
if "index" not in st.session_state:
    st.session_state.index = None
if "filters" not in st.session_state:
    st.session_state.filters = None
if "messages" not in st.session_state:
    st.session_state.messages = []  # Chat history

# Store results into state
if pdf:
    index, filters = load_pdf(pdf)
    st.session_state.index = index       # Persists across reruns!
    st.session_state.filters = filters

# Read from state later
if st.session_state.index:
    query = st.text_input("Ask a question:")
    if st.button("Submit") and query:
        response = query_db(st.session_state.index, st.session_state.filters, query)
        st.write(response)
```

> [!CAUTION]
> **Never** store large objects like the `VectorStoreIndex` directly in session_state if they can't be pickled. Store the doc_id/filters and rebuild the index from ChromaDB instead.

---

## Step 5 — Caching (Avoid Re-running Expensive Functions)

```python
# Cache a function so it only re-runs when inputs change
@st.cache_resource          # For heavy objects: models, DB connections
def init_ocr_reader():
    return easyocr.Reader(['en'], gpu=True)

@st.cache_data              # For data: DataFrames, text, embeddings
def load_document(doc_id: str):
    # Only re-runs if doc_id changes
    return load_pdf(doc_id)

# Usage
reader = init_ocr_reader()  # Loads once, cached forever
```

| Decorator | Use for |
|---|---|
| `@st.cache_resource` | Models, DB clients, OCR readers — shared objects |
| `@st.cache_data` | Data results — pickled, per-input cached |

---

## Step 6 — Streaming Output

Streamlit can stream LLM responses token by token:

```python
# Method 1: st.write_stream (simplest)
if query:
    response = query_db(index, filters, query)
    st.write_stream(response.response_gen)  # Streams tokens live

# Method 2: Manual streaming with a placeholder
placeholder = st.empty()
full_text = ""
for token in response.response_gen:
    full_text += token
    placeholder.markdown(full_text + "▌")  # Blinking cursor effect
placeholder.markdown(full_text)            # Final text without cursor
```

---

## Step 7 — Progress & Spinners

```python
# Spinner — show while a long task runs
with st.spinner("Embedding document..."):
    index, filters = load_pdf(uploaded_file)
st.success("Done!")

# Progress bar
progress = st.progress(0)
for i, page in enumerate(pages):
    process(page)
    progress.progress((i + 1) / len(pages))

# Status container (expandable live log)
with st.status("Processing PDF...", expanded=True) as status:
    st.write("Extracting text...")
    text = extract_pdf_text(path)
    st.write("Embedding chunks...")
    index = embed(text)
    status.update(label="Complete!", state="complete")
```

---

## Step 8 — Putting It Together (Your App Pattern)

```python
import streamlit as st
from backend import load_pdf, website_link, query_db

st.set_page_config(page_title="Legal TL;DR", page_icon="⚖️", layout="wide")

# --- Initialize session state ---
if "index" not in st.session_state:
    st.session_state.index = None
if "filters" not in st.session_state:
    st.session_state.filters = None

# --- Sidebar: Document Input ---
with st.sidebar:
    st.header("📄 Load Document")
    mode = st.radio("Input type", ["PDF Upload", "URL"])

    if mode == "PDF Upload":
        pdf = st.file_uploader("Upload PDF", type=["pdf"])
        if pdf and st.button("Process PDF"):
            with st.spinner("Processing..."):
                idx, flt = load_pdf(pdf)
                st.session_state.index = idx
                st.session_state.filters = flt
            st.success("Ready!")
    else:
        url = st.text_input("Enter URL")
        if url and st.button("Process URL"):
            with st.spinner("Scraping & embedding..."):
                idx, flt = website_link(url)
                st.session_state.index = idx
                st.session_state.filters = flt
            st.success("Ready!")

# --- Main: Query Interface ---
st.title("⚖️ Legal TL;DR")

if st.session_state.index:
    query = st.text_input("Ask a question about the document:")
    if st.button("Submit") and query:
        with st.spinner("Thinking..."):
            response = query_db(st.session_state.index, st.session_state.filters, query)
        st.markdown("### Answer")
        st.write_stream(response.response_gen)
        with st.expander("📚 Sources"):
            for i, node in enumerate(response.source_nodes):
                st.markdown(f"**[{i+1}]** _{node.score:.3f}_ — {node.node.get_content()[:300]}...")
else:
    st.info("👈 Load a document from the sidebar to get started.")
```

---

## Quick Reference

| Component | Code |
|---|---|
| Title | `st.title()` |
| Text | `st.write()`, `st.markdown()` |
| Alerts | `st.info()`, `st.success()`, `st.warning()`, `st.error()` |
| Input | `st.text_input()`, `st.text_area()`, `st.file_uploader()` |
| Button | `st.button()` |
| Layout | `st.sidebar`, `st.columns()`, `st.tabs()`, `st.expander()` |
| State | `st.session_state["key"]` |
| Cache | `@st.cache_resource`, `@st.cache_data` |
| Loading | `st.spinner()`, `st.progress()`, `st.status()` |
| Stream | `st.write_stream()`, `st.empty()` |
| Rerun | `st.rerun()` |
