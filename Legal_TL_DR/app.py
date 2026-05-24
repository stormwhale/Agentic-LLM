import streamlit as st
import backend

st.set_page_config(
    page_title="Legal Document Analyzer (TL:DR)",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ Legal Document Analyzer (TL:DR)")
st.subheader("This app is designed to store and extract key information from legal documents.")
st.write("You can also paste web site page links on the sidebar or upload pdf locally.")

#Define a progress bar:
def update_progress(percent, text):
    progress_bar.progress(percent, text=text)

#Create a sidebar:
with st.sidebar:
    #Create navigation in sidebar:
    page_selection = st.radio("Navigation", ['Chat Box', 'Library Browser'])

    st.divider()
    st.write("Document processing section:")

    st.header("📄 Document Uploader")
    uploader = st.file_uploader("Upload PDF", type=['pdf'])
    if st.button("Upload PDF"):
        if uploader is not None:
            progress_bar = st.progress(0, text="Initializing...")
            index, filters, exist = backend.load_pdf(uploader, progress_callback=update_progress)
            st.session_state['index'] = index
            st.session_state['filters'] = filters
            
            if exist == True:
                st.session_state['upload_status'] = ("info", "PDF already exists in the database.")
            else:
                st.session_state['upload_status'] = ("success", "PDF embedded to DB successfully.")            
            st.rerun()
        else:
            st.warning("Please upload a PDF file first.")

    # Display status message if it exists in session state
    if 'upload_status' in st.session_state:
        status_type, message = st.session_state.pop('upload_status')
        if status_type == "info":
            st.info(message)
        elif status_type == "success":
            st.success(message)

    #URL_input_section:
    st.header("🌐 URL Link")
    url_input = st.text_input("Paste URL Here:")
    if st.button("Parse Website"):
        if url_input.strip():
            progress_bar = st.progress(0, text="Initializing...")
            index, filters, exist = backend.website_link(url_input, progress_callback=update_progress)
            st.session_state['index'] = index
            st.session_state['filters'] = filters

            if exist == True:
                st.session_state['website_status'] = ("info", "Website already exists in the database.")
            else:
                st.session_state['website_status'] = ("success", "Website embedded to DB successfully.")            
            st.rerun()
        else:
            st.warning("Please enter a URL first.")
    
    #Display status message if it exists in session state
    if 'website_status' in st.session_state:
        status_type, message = st.session_state.pop('website_status')
        if status_type == "info":
            st.info(message)
        elif status_type == "success":
            st.success(message)

#=====================================================================================
#Main chat section UI:
#Idea 1 split the chat window into 2- one for chat and one for sources.
#=====================================================================================
#Chat widget (chat-box)
#=====================================================================================
if page_selection == 'Chat Box':
    chat_col, sources_col = st.columns([2, 1])

    with chat_col:
        st.header("Main Chat")
        #Create toggle to switch between general and specific mode:
        option_map = {
            0: "Selected Documents",
            1: "Entire Database"
        }

        search_mode = st.pills(
            "Search Mode Selection",
            options=option_map.keys(),            
            format_func=lambda option: option_map[option],
            selection_mode="single"
        )

        selected_filters = None
        if search_mode == 0: #Selected Documents
            all_docs = backend.get_all_registry_docs()

            if all_docs:
                selected_labels = st.multiselect(
                    "Select documents to search:",
                    options=[d["label"] for d in all_docs],
                    default=[]
                )

                selected_ids = [
                    d["doc_id"] for d in all_docs if d["label"] in selected_labels
                ]
                selected_filters = backend.build_filters(selected_ids)
        # In "Entire Database" mode, no document selector is shown — search runs across all docs

        with st.container(height=700):
            st.write("Ask anything about the document and I'll give you a TL;DR.")

            if "messages" not in st.session_state:
                with st.chat_message("assistant"):
                    st.write("Hi, I'm Legal TL;DR Assistant. You can ask me anything about the embedded documents.")
            st.session_state.messages = [
                {"role": 'system',
                "content": (
                    f"Today is {backend.get_current_time()}. \n\n",
                    'You are a very accurate TL;DR assistant.',
                    'Always answer in a TL:DR format.',
                    'Only cite sources using the exact [citations] references provided to you. Do not invent citation numbers.',
                    'If you do not find the answer, say "No such information is available in the database."'
                )}
            ]

            for message in st.session_state.messages:
                if message['role'] != 'system':
                    with st.chat_message(message['role']):
                        st.markdown(message['content'])

            #Get user input:
            if prompt := st.chat_input("Ask ME anything about the documents."):
                st.session_state.messages.append({'role': 'user', 'content': prompt})
                with st.chat_message('user'):
                    st.markdown(prompt)

                #Get LLM response:
                with st.chat_message('assistant'):
                    message_placeholder = st.empty()

                    with st.spinner('🕵️‍♂️ Fetching and thinking...🤔'):
                        st.session_state.pop('sources', None)  # Clear stale citations before new query
                        if 'index' not in st.session_state:
                            st.session_state['index'] = backend.VectorStoreIndex.from_vector_store(vector_store=backend.embeddings_vector_store)
                        current_mode = "specific" if search_mode == 0 else "general"
                        response = backend.query_db(
                            st.session_state['index'],
                            filters=selected_filters,
                            query=prompt
                        )

                        full_response = ""
                        for chunk in response.response_gen:
                            full_response += chunk
                            message_placeholder.markdown(full_response + "▌")
                        #Remove cursor and output final response
                        message_placeholder.markdown(full_response)
                        #Save the response source nodes to session state:
                        st.session_state['sources'] = response.source_nodes
                    #Save the LLM's message to the session state:
                    st.session_state.messages.append({'role': 'assistant', 'content': full_response})
    
    #Sources section:
        with sources_col:
            st.header("📑 Retrieved Evidence")
            st.caption("Context chunks the LLM was given to form its answer, ranked by relevance.")
            if "sources" in st.session_state and st.session_state['sources']:
                # Sort by relevance score (highest first)
                sorted_sources = sorted(
                    st.session_state['sources'],
                    key=lambda s: s.score if s.score is not None else 0,
                    reverse=True
                )
                for num, source in enumerate(sorted_sources, 1):
                    score = source.score
                    score_label = f" | Relevance: {score:.2%}" if score is not None else ""
                    with st.expander(f"Source {num}: {source.metadata.get('source', 'Unknown')}{score_label}"):
                        st.write(source.text)
            else:
                st.info("Ask a question to show sources.")

#=====================================================================================
#ChromaDB stats for the browser tab:
#=====================================================================================
Total_docs = backend.chroma_registry_collection.count()

#Define a function to count the number of documents in the database:
def doc_counter():
    registry_metadata =backend.chroma_registry_collection.get(include=['metadatas'])
    pdf_count = 0
    website_count = 0
    for i in registry_metadata['metadatas']:
        if i['type'].startswith('pdf'):
            pdf_count += 1
        elif i['type'] == "website":
            website_count += 1
    return pdf_count, website_count

pdf_count, website_count = doc_counter()

#=====================================================================================
#Library Browser UI:
#=====================================================================================
if page_selection == 'Library Browser':    
    st.header("📚 Library Browser")
    st.write("All the embedded documents are listed here.")

    #Stat-shows total number of documents in the database.
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Documents Saved", Total_docs)
    col2.metric("Total PDFs", pdf_count)
    col3.metric("Total Websites", website_count)

    st.divider()

    #Add a "Show all" button:
    if st.button("🤓 Show all documents"):
        st.session_state['search_results'] = backend.chroma_registry_collection.get(
            limit = Total_docs
        )

    #Filter Bar:
    filter_col, num_result_col, search_col, search_but= st.columns([1, 1, 3, 1])
    with filter_col:
        source_filter = st.selectbox("Filter by", ["All", "pdf_text", "pdf_ocr", "website"])

    with num_result_col:
        num_results = st.selectbox("Number of results", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    with search_col:
        search_term = st.text_input("Search documents...")
        term = search_term.lower().strip()
    
    with search_but:
        if st.button("Filter"):
            # Prepare arguments. If 'All' is selected, don't filter by type.
            query_kwargs = {}
            if source_filter != "All":
                query_kwargs["where"] = {"type": source_filter}
                
            # If user typed something, do a similarity search
            if term == "":
                st.info("Please enter a search term or enter 'show all' to list all the documents.")
            
            elif term == "show all":
                st.session_state['search_results'] = backend.chroma_registry_collection.get(
                    limit=Total_docs, **query_kwargs
                )
            else:
                st.session_state['search_results'] = backend.chroma_registry_collection.query(
                    query_texts=[term],
                    n_results=num_results,
                    **query_kwargs
                )
    

    # Initialize empty results if nothing has been searched yet
    if 'search_results' not in st.session_state:
        st.session_state['search_results'] = {"metadatas": []}

    result = st.session_state['search_results']

    #Normalize search results for .get() and .query():
    raw_metadatas = result.get('metadatas', [])
    raw_documents = result.get('documents', [])

    flat_metadatas = []
    if raw_metadatas:
        if isinstance(raw_metadatas[0], list):
            flat_metadatas = raw_metadatas[0]
        else:
            flat_metadatas = raw_metadatas
    
    flat_documents = []
    if raw_documents:
        if isinstance(raw_documents[0], list):
            flat_documents = raw_documents[0]
        else:
            flat_documents = raw_documents
    
    #Main panel:
    list_col, preview_col = st.columns([1, 2])
    with list_col:
        st.subheader("Documents")

        if flat_metadatas:
            indices = list(range(len(flat_metadatas)))

            selected_idx = st.radio(
                "Select a document",
                indices,
                format_func=lambda i: backend.truncate_30(flat_metadatas[i].get('source', 'Unknown'))
            )
        else:
            st.write("No documents found. Click 'Filter' to search.")
            selected_idx = None

    with preview_col:
        if selected_idx is not None:
            selected_metadata = flat_metadatas[selected_idx]
            summary_text ="No summary available"
            if flat_documents and len(flat_documents) > selected_idx:
                summary_text = flat_documents[selected_idx]
            
            #UI preview:
            st.subheader(f"Preview")
            st.write(f"Title: {selected_metadata.get('source', 'Unknown')}")
            st.write(f"Type: {selected_metadata.get('type', 'Unknown')}")
            st.write(f"Doc. ID: {selected_metadata.get('doc_id', 'Unknown')}")

            st.divider()

            st.write("Summary/TL:DR")
            st.info(f"**Summary:** {summary_text}")

            #Create a delete button:        
            if st.button("Delete Selected Document"):
                doc_id = selected_metadata.get("doc_id")
                backend.chroma_registry_collection.delete(ids=[doc_id])
                backend.chroma_embeddings_collection.delete(where={"document_id": doc_id})
                
                # Remove only the deleted document from the session state search results
                if 'search_results' in st.session_state:
                    res = st.session_state['search_results']
                    for key in ['metadatas', 'documents', 'ids']:
                        if res.get(key) is not None and len(res[key]) > 0:
                            if isinstance(res[key][0], list):
                                if len(res[key][0]) > selected_idx:
                                    res[key][0].pop(selected_idx)
                            else:
                                if len(res[key]) > selected_idx:
                                    res[key].pop(selected_idx)
                
                st.rerun()
                       
        else:
            st.info("👈 Select a document to view the preview.")