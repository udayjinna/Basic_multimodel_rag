import streamlit as st
import fitz  # PyMuPDF
from langchain_core.documents import Document
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np
from langchain.schema.messages import HumanMessage
from sklearn.metrics.pairwise import cosine_similarity
import os
import base64
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
import tempfile

# Page configuration
st.set_page_config(
    page_title="Multimodal PDF RAG Assistant",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'image_data_store' not in st.session_state:
    st.session_state.image_data_store = {}
if 'all_docs' not in st.session_state:
    st.session_state.all_docs = []
if 'clip_model' not in st.session_state:
    st.session_state.clip_model = None
if 'clip_processor' not in st.session_state:
    st.session_state.clip_processor = None

# Title and description
st.title("📚 Multimodal PDF RAG Assistant")
st.markdown("""
This app uses CLIP embeddings and Gemini to answer questions about your PDF documents, 
including both text and images!
""")

# Sidebar for API keys and settings
with st.sidebar:
    st.header("⚙️ Configuration")
    
    google_api_key = st.text_input(
        "Google API Key (Gemini)",
        type="password",
        help="Enter your Google API key for Gemini"
    )
    
    hf_token = st.text_input(
        "HuggingFace Token (Optional)",
        type="password",
        help="Optional: For accessing gated models"
    )
    
    st.markdown("---")
    st.header("📄 Upload PDF")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload a PDF document to analyze"
    )
    
    k_results = st.slider(
        "Number of results to retrieve",
        min_value=1,
        max_value=10,
        value=5,
        help="How many relevant chunks to retrieve for context"
    )

# Helper functions
@st.cache_resource
def load_clip_model():
    """Load CLIP model and processor"""
    with st.spinner("Loading CLIP model..."):
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model.eval()
    return model, processor

def embed_image(image_data, model, processor):
    """Embed image using CLIP"""
    if isinstance(image_data, str):
        image = Image.open(image_data).convert("RGB")
    else:
        image = image_data

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()

def embed_text(text, model, processor):
    """Embed text using CLIP"""
    inputs = processor(
        text=text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    )
    with torch.no_grad():
        features = model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()

def process_pdf(pdf_file, model, processor):
    """Process PDF and extract text and images"""
    all_docs = []
    all_embeddings = []
    image_data_store = {}
    
    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_file.read())
        tmp_path = tmp_file.name
    
    doc = fitz.open(tmp_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_pages = len(doc)
    
    for i, page in enumerate(doc):
        status_text.text(f"Processing page {i+1}/{total_pages}...")
        progress_bar.progress((i + 1) / total_pages)
        
        # Process text
        text = page.get_text()
        if text.strip():
            temp_doc = Document(page_content=text, metadata={"page": i, "type": "text"})
            text_chunks = splitter.split_documents([temp_doc])
            
            for chunk in text_chunks:
                embedding = embed_text(chunk.page_content, model, processor)
                all_embeddings.append(embedding)
                all_docs.append(chunk)
        
        # Process images
        for img_index, img in enumerate(page.get_images(full=True)):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                image_id = f"page_{i}_img_{img_index}"
                
                buffered = io.BytesIO()
                pil_image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                image_data_store[image_id] = img_base64
                
                embedding = embed_image(pil_image, model, processor)
                all_embeddings.append(embedding)
                
                image_doc = Document(
                    page_content=f"[Image: {image_id}]",
                    metadata={"page": i, "type": "image", "image_id": image_id}
                )
                all_docs.append(image_doc)
                
            except Exception as e:
                st.warning(f"Error processing image {img_index} on page {i}: {e}")
                continue
    
    doc.close()
    os.unlink(tmp_path)
    
    progress_bar.empty()
    status_text.empty()
    
    return all_docs, all_embeddings, image_data_store

def create_vector_store(all_docs, embeddings_array):
    """Create FAISS vector store"""
    vector_store = FAISS.from_embeddings(
        text_embeddings=[(doc.page_content, emb) for doc, emb in zip(all_docs, embeddings_array)],
        embedding=None,
        metadatas=[doc.metadata for doc in all_docs]
    )
    return vector_store

def retrieve_multimodal(query, vector_store, model, processor, k=5):
    """Unified retrieval using CLIP embeddings"""
    query_embedding = embed_text(query, model, processor)
    results = vector_store.similarity_search_by_vector(
        embedding=query_embedding,
        k=k
    )
    return results

def create_multimodal_message(query, retrieved_docs, image_data_store):
    """Create a message with both text and images"""
    content = []
    
    content.append({
        "type": "text",
        "text": f"Question: {query}\n\nContext:\n"
    })
    
    text_docs = [doc for doc in retrieved_docs if doc.metadata.get("type") == "text"]
    image_docs = [doc for doc in retrieved_docs if doc.metadata.get("type") == "image"]
    
    if text_docs:
        text_context = "\n\n".join([
            f"[Page {doc.metadata['page']}]: {doc.page_content}"
            for doc in text_docs
        ])
        content.append({
            "type": "text",
            "text": f"Text excerpts:\n{text_context}\n"
        })
    
    for doc in image_docs:
        image_id = doc.metadata.get("image_id")
        if image_id and image_id in image_data_store:
            content.append({
                "type": "text",
                "text": f"\n[Image from page {doc.metadata['page']}]:\n"
            })
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_data_store[image_id]}"
                }
            })
    
    content.append({
        "type": "text",
        "text": "\n\nPlease answer the question based on the provided text and images."
    })
    
    return HumanMessage(content=content)

# Main application logic
if uploaded_file and google_api_key:
    
    # Load CLIP model
    if st.session_state.clip_model is None:
        st.session_state.clip_model, st.session_state.clip_processor = load_clip_model()
    
    # Process PDF button
    if not st.session_state.processed:
        if st.button("🔄 Process PDF", type="primary"):
            with st.spinner("Processing PDF... This may take a few minutes."):
                try:
                    all_docs, all_embeddings, image_data_store = process_pdf(
                        uploaded_file,
                        st.session_state.clip_model,
                        st.session_state.clip_processor
                    )
                    
                    embeddings_array = np.array(all_embeddings)
                    vector_store = create_vector_store(all_docs, embeddings_array)
                    
                    st.session_state.all_docs = all_docs
                    st.session_state.vector_store = vector_store
                    st.session_state.image_data_store = image_data_store
                    st.session_state.processed = True
                    
                    st.success(f"✅ Successfully processed {len(all_docs)} document chunks!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
    
    # Query interface
    if st.session_state.processed:
        st.markdown("---")
        st.header("💬 Ask Questions")
        
        query = st.text_area(
            "Enter your question:",
            height=100,
            placeholder="e.g., What is the main topic discussed in this document?"
        )
        
        if st.button("🔍 Get Answer", type="primary"):
            if query.strip():
                with st.spinner("Analyzing and generating answer..."):
                    try:
                        # Initialize LLM
                        llm = ChatGoogleGenerativeAI(
                            model="gemini-2.5-flash",
                            google_api_key=google_api_key,
                            temperature=0
                        )
                        
                        # Retrieve relevant documents
                        context_docs = retrieve_multimodal(
                            query,
                            st.session_state.vector_store,
                            st.session_state.clip_model,
                            st.session_state.clip_processor,
                            k=k_results
                        )
                        
                        # Show retrieved context
                        with st.expander("📋 Retrieved Context", expanded=False):
                            for doc in context_docs:
                                doc_type = doc.metadata.get("type", "unknown")
                                page = doc.metadata.get("page", "?")
                                if doc_type == "text":
                                    st.markdown(f"**Text from page {page}:**")
                                    st.text(doc.page_content[:200] + "...")
                                else:
                                    st.markdown(f"**Image from page {page}**")
                        
                        # Create message and get response
                        message = create_multimodal_message(
                            query,
                            context_docs,
                            st.session_state.image_data_store
                        )
                        response = llm.invoke([message])
                        
                        # Display answer
                        st.markdown("### 📝 Answer:")
                        st.markdown(response.content)
                        
                    except Exception as e:
                        st.error(f"Error generating answer: {str(e)}")
            else:
                st.warning("Please enter a question.")
        
        # Reset button
        if st.button("🔄 Process New PDF"):
            st.session_state.processed = False
            st.session_state.vector_store = None
            st.session_state.image_data_store = {}
            st.session_state.all_docs = []
            st.rerun()

elif not uploaded_file:
    st.info("👈 Please upload a PDF file using the sidebar to get started.")
elif not google_api_key:
    st.warning("⚠️ Please enter your Google API Key in the sidebar to continue.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with Streamlit, CLIP, and Gemini | 
    <a href='https://github.com' target='_blank'>GitHub</a></p>
</div>
""", unsafe_allow_html=True)
