import time

# Time imports
t0 = time.time()
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
import chromadb
t1 = time.time()
print(f"1. Imports: {t1-t0:.1f}s")

# Time LLM/embed setup
llm = Ollama(model="qwen3.5:4b", request_timeout=300.0, num_gpu=99, context_window=4096,
             additional_kwargs={"think": False})
Settings.llm = llm
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text:latest", base_url="http://localhost:11434")
t2 = time.time()
print(f"2. LLM/Embed init: {t2-t1:.1f}s")

# Time ChromaDB + index
chroma_client = chromadb.PersistentClient(path="./chroma_db")
col = chroma_client.get_or_create_collection("doc_embeddings")
print(f"   Chunks in DB: {col.count()}")
vs = ChromaVectorStore(chroma_collection=col)
t3 = time.time()
print(f"3. ChromaDB load: {t3-t2:.1f}s")

index = VectorStoreIndex.from_vector_store(vector_store=vs)
t4 = time.time()
print(f"4. Index build: {t4-t3:.1f}s")

# Time query
query_engine = index.as_query_engine(filters=None, similarity_top_k=3, streaming=True)
t5 = time.time()
print(f"5. Query engine create: {t5-t4:.1f}s")

response = query_engine.query("What is the document about?")
t6 = time.time()
print(f"6. query() call (retrieval + LLM prompt): {t6-t5:.1f}s")

# Time streaming
full = ""
for chunk in response.response_gen:
    full += chunk
t7 = time.time()
print(f"7. Stream all tokens: {t7-t6:.1f}s")
print(f"\nTotal: {t7-t0:.1f}s")
print(f"Response length: {len(full)} chars")
print(f"Response preview: {full[:200]}")
