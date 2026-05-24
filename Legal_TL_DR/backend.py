from llama_index.core import VectorStoreIndex, Settings, StorageContext, Document
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.llms.ollama import Ollama
from llama_index.readers.web import BeautifulSoupWebReader
from llama_index.embeddings.ollama import OllamaEmbedding
import hashlib
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterCondition
from llama_index.core.query_engine import CitationQueryEngine
from PIL import Image
import numpy as np
import easyocr, fitz
import datetime as dt

llm = Ollama(model="qwen2.5:7b",
             request_timeout=300.0,
             num_gpu=99,              # Force all layers onto GPU
             context_window=4096,
             additional_kwargs={"think": False})  # Disable qwen3 chain-of-thought (major speedup)

# Configure global settings
Settings.llm = llm
Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text:latest",
    base_url="http://localhost:11434",
)

# Define chroma vector storage:
chroma_client = chromadb.PersistentClient(path="./chroma_db")

#Collection 1: Data registry. This is like a catalog for all the docs. It will store the metadata of all the docs.
chroma_registry_collection = chroma_client.get_or_create_collection("doc_registry")
registry_vector_store = ChromaVectorStore(chroma_collection=chroma_registry_collection)

#Collection 2: Main corpus. This is where the actual embeddings of the docs are stored.
chroma_embeddings_collection = chroma_client.get_or_create_collection("doc_embeddings")
embeddings_vector_store = ChromaVectorStore(chroma_collection=chroma_embeddings_collection)

#Modularize the functions:
#Def a hash function to get unique IDs:
def get_hash_id(text:str):
  return hashlib.sha256(text.encode('utf-8')).hexdigest()

#Define a function for time:
def get_current_time():
  return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#Create a function to embed a website:
def website_link(target_url:str, progress_callback=None):
  '''This function will take a url and check against the db and embed if not found'''
  #Create unique hash ID:
  doc_id = get_hash_id(target_url)

  #Check for existing doc in DB:
  existing_docs = chroma_registry_collection.get(ids=[doc_id])

  #Create a signal tag to flag whether the website was already processed:
  exist = False

  if progress_callback: progress_callback(10, "Checking ChromaDB for existing entry...")

  if len(existing_docs['ids']) > 0:
    print('Document exists in DB. Retrieving...')
    filters = MetadataFilters(filters=[
      MetadataFilter(key="document_id", value=doc_id)
    ])
    index = VectorStoreIndex.from_vector_store(vector_store=embeddings_vector_store)
    exist = True

    return index, filters, exist
  else:
    if progress_callback: progress_callback(20, "No existing entry found. Initializing embedding process...")
    print('Embedding new document...')
    #Step 1 scrape and embed to the main storage:
    docs = BeautifulSoupWebReader().load_data([target_url])
    for doc in docs:
      doc.doc_id = doc_id
      doc.metadata["document_id"] = doc_id
    # Save the embeddings:
    storage_context = StorageContext.from_defaults(vector_store=embeddings_vector_store)
    index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)

    #Step 2 save the metadata to the registry:
    if progress_callback: progress_callback(40, "Summarizing website for catalog...")
    full_text = " ".join([doc.text for doc in docs])[:3000]
    summary_prompt = f"Summarize this document in 2-3 sentences for a document catalog:\n\n{full_text}"
    summary = str(llm.complete(summary_prompt))

    #Step 3: Store one entry in the registry:
    if progress_callback: progress_callback(80, "Finalizing and saving to DB...")
    chroma_registry_collection.add(
      ids=[doc_id],
      documents=[summary],
      metadatas=[{
        "source": target_url,
        "type": "website",
        "doc_id": doc_id
      }]
    )
    if progress_callback: progress_callback(100, "Embedding complete. Ready to query!")
    filters = MetadataFilters(filters=[
      MetadataFilter(key="document_id", value=doc_id)
    ])
    return index, filters, exist

#Define a text extractor:
# EasyOCR reader is initialized once at module level to avoid reloading the model on every call.
ocr_reader = easyocr.Reader(['en'], gpu=True)

def extract_pdf_text(pdf_file: str, progress_callback=None) -> tuple[str, bool]:
  '''
  Stage 1: Try native text extraction (fast, no GPU needed).
  Stage 2: Fall back to EasyOCR on GPU if yield is too low.
  Returns (text, ocr_used).
  '''
  if progress_callback: progress_callback(10, "Opening PDF...")
  # Handle Streamlit UploadedFile (file-like object) vs string path
  if hasattr(pdf_file, "read"):
      pdf_file.seek(0)
      doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
      pdf_file.seek(0) # Reset for potential future use
  else:
      doc = fitz.open(pdf_file)

  # Fast path: native selectable text
  if progress_callback: progress_callback(20, "Attempting native text extraction...")
  text = "\n".join(page.get_text() for page in doc)

  if len(text.strip()) >= 100:
    doc.close()
    print(f"Native extraction OK ({len(text)} chars)")
    if progress_callback: progress_callback(100, "Native extraction successful.")
    return text, False

  # EasyOCR GPU fallback:
  print("Native extraction weak. Switching to EasyOCR (GPU)...")
  if progress_callback: progress_callback(30, "Native extraction weak. Initializing OCR...")
  images = []

  # Convert all PDF pages to PIL images first:
  for page in doc:
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 144 DPI
    images.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))

  doc.close()

  # Run OCR on all pages:
  ocr_pages = []
  total_pages = len(images)
  for i, img in enumerate(images):
    current_page = i + 1
    if progress_callback:
        # Scale progress from 40% to 90% during OCR
        percent = 40 + int((i / total_pages) * 50)
        progress_callback(percent, f"OCR: Processing page {current_page}/{total_pages}...")
    
    result = ocr_reader.readtext(np.array(img), detail=0, paragraph=True)
    ocr_pages.append(" ".join(result))
    print(f"  OCR: page {current_page}/{total_pages}")

  text = "\n".join(ocr_pages)

  if len(text.strip()) < 100:
    raise RuntimeError("OCR ran but still returned low text yield. PDF may be unreadable.")

  print(f"EasyOCR complete ({len(text)} chars)")
  if progress_callback: progress_callback(95, "OCR complete. Finalizing text...")
  return text, True
    


#Define a function to parse or check locally uploaded PDF:

def load_pdf(pdf_file: str, progress_callback=None):
  '''
  Load a PDF, automatically try to fast extract. If it fails, OCR is enabled for text extraction.
  Then embed it, and return an index and filters.
  '''
  #Create signal flag:
  exist = False

  #Check if PDF is image or text:
  text, ocr = extract_pdf_text(pdf_file, progress_callback=progress_callback)
   
  # Use content-based hashing for better duplicate detection
  mode_tag = ":ocr" if ocr else ":text"
  
  if hasattr(pdf_file, "read"):
      pdf_file.seek(0)
      content = pdf_file.read()
      pdf_file.seek(0) # Reset for fitz
      # Hash the bytes directly + mode tag
      doc_id = hashlib.sha256(content + mode_tag.encode()).hexdigest()
  else:
      # Fallback for string paths
      doc_id = get_hash_id(str(pdf_file) + mode_tag)

  #Check for existing doc in DB registry:
  existing_docs = chroma_registry_collection.get(ids=[doc_id])

  if len(existing_docs['ids']) > 0:
    print('Document exists in DB. Retrieving...')
    filters = MetadataFilters(filters=[
      MetadataFilter(key="document_id", value=doc_id)
    ])
    index = VectorStoreIndex.from_vector_store(vector_store=embeddings_vector_store)
    exist = True
    return index, filters, exist
  else:
    # Use the filename for storage metadata
    file_name = pdf_file.name if hasattr(pdf_file, 'name') else str(pdf_file)

    print(f'Embedding new document (ocr={ocr})...')
    docs = [Document(
      text=text,
      doc_id=doc_id,
      metadata={"document_id": doc_id, "source": file_name, "ocr": ocr})]
      
    if progress_callback: progress_callback(96, "Creating embeddings (this may take a moment)...")
    # Save the embeddings to the embeddings vector store:
    storage_context = StorageContext.from_defaults(vector_store=embeddings_vector_store)
    index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)
    print('Embedding completed.')

    # Save the metadata to the registry vector store:
    full_text = " ".join([doc.text for doc in docs])[:3000]
    summary_prompt = f"Summarize this document in 2-3 sentences for a document catalog:\n\n{full_text}"
    if progress_callback: progress_callback(97, "Generating summary with LLM...")
    summary = str(llm.complete(summary_prompt))
    if progress_callback: progress_callback(100, "Processing complete!")

    #Add the summary to the registry:
    chroma_registry_collection.add(
      ids=[doc_id],
      documents=[summary],
      metadatas=[{
        "source": file_name,
        "type": f"pdf_{'ocr' if ocr else 'text'}",
        "doc_id":doc_id
      }]
    )
    filters = MetadataFilters(filters=[
      MetadataFilter(key="document_id", value=doc_id)
    ])
    
    return index, filters, exist
  
#Define a function for querying the DB:
def build_filters(doc_ids: list[str]):
  if not doc_ids:
    return None
  if len(doc_ids) == 1: #Only one document is needed
    return MetadataFilters(filters=[
      MetadataFilter(key="document_id", value=doc_ids[0])
    ])
  #This is for multiple documents in doc_ids:
  return MetadataFilters(
    filters=[MetadataFilter(key="document_id", value=d_id) for d_id in doc_ids], condition=FilterCondition.OR
  )

def get_all_registry_docs():
  """Fetch all docs from registry for the mutliselect UI."""
  result = chroma_registry_collection.get(include=['metadatas'])
  docs = []
  for meta in result['metadatas']:
    docs.append({
      "label": truncate_30(meta.get('source', 'Unknown source')),
      "doc_id": meta.get("doc_id"),
      "type": meta.get("type")
    })
  return docs

def query_db(index, filters, query: str):
  """
  A function for streamlit to call to get LLM responses/
  """
  query_engine = CitationQueryEngine.from_args(
      index,
      filters=filters,
      similarity_top_k=5,
      citation_chunk_size=512,
      streaming=True
    ) 

  response = query_engine.query(query)
  return response
  


'''def query_db(index, filters, query:str):
  This function will take an index and query and return the response
  query_engine = CitationQueryEngine.from_args(
    index,
    filters=filters,
    similarity_top_k=4,
    citation_chunk_size=512,
    streaming=True
  )
  print('Generating answers...')
  response = query_engine.query(query)
  return response'''

def truncate_30(text:str):
  if len(text) > 30:
    return str(text)[:30] + "..."
  else:
    return str(text)


