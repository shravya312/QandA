import os
import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import google.generativeai as genai
from dotenv import load_dotenv
import tempfile
import shutil
import numpy as np

# ===== Load Environment Variables =====
load_dotenv()

# ===== Configuration =====
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Qdrant client
try:
    qdrant_client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )
except Exception as e:
    st.error(f"Failed to initialize Qdrant client: {str(e)}")
    st.stop()

# Initialize sentence transformer model for embeddings
try:
    model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    st.error(f"Failed to initialize SentenceTransformer: {str(e)}")
    st.stop()

# Configure Gemini API
try:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    st.error(f"Failed to initialize Gemini API: {str(e)}")
    st.stop()

# Constants
COLLECTION_NAME = "pdf_collection"
VECTOR_SIZE = 384
CHUNK_SIZE = 1000

# ===== Core Functions =====
def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF"""
    try:
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None

def chunk_text(text):
    """Split text into manageable chunks"""
    if not text:
        return []
    return [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE) if text[i:i + CHUNK_SIZE].strip()]

def get_embeddings(chunks):
    """Convert text chunks into embeddings"""
    try:
        return model.encode(chunks)
    except Exception as e:
        st.error(f"Error generating embeddings: {str(e)}")
        return None

def setup_collection():
    """Recreate Qdrant collection"""
    try:
        qdrant_client.delete_collection(COLLECTION_NAME)
    except Exception as e:
        st.warning(f"Could not delete existing collection: {str(e)}")
    
    try:
        qdrant_client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )
        return True
    except Exception as e:
        st.error(f"Error setting up collection: {str(e)}")
        return False

def upload_to_qdrant(chunks, embeddings):
    """Upload data to Qdrant"""
    if not chunks or embeddings is None or (isinstance(embeddings, np.ndarray) and embeddings.size == 0) or (hasattr(embeddings, '__len__') and len(embeddings) == 0):
        return False
    
    try:
        points = [
            PointStruct(id=i, vector=embedding, payload={"text": chunk})
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        return True
    except Exception as e:
        st.error(f"Error uploading to Qdrant: {str(e)}")
        return False

def search_chunks(query_vector):
    """Retrieve most relevant chunks"""
    try:
        result = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=5
        )
        return [hit.payload["text"] for hit in result]
    except Exception as e:
        st.error(f"Error searching chunks: {str(e)}")
        return []

def generate_answer_from_gemini(query, context):
    """Use Gemini to answer the question"""
    if not query or not context:
        return "Please provide both a question and context."
    
    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Error from Gemini: {str(e)}"

# ===== Streamlit UI =====
st.set_page_config(layout="wide")
st.title("📄 PDF Question Answering System with RAG")

# Initialize session state
if 'processed_pdf' not in st.session_state:
    st.session_state.processed_pdf = False
if 'pdf_path' not in st.session_state:
    st.session_state.pdf_path = None

pdf = st.file_uploader("Upload a PDF file", type=["pdf"])
question = st.text_input("Ask a question based on the PDF content:")

if pdf and not st.session_state.processed_pdf:
    # Create a temporary file to store the PDF
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, "uploaded.pdf")
    
    try:
        with open(temp_path, "wb") as f:
            f.write(pdf.read())
        
        st.session_state.pdf_path = temp_path
        st.info("⏳ Extracting and processing PDF...")
        
        text = extract_text_from_pdf(temp_path)
        if text:
            chunks = chunk_text(text)
            embeddings = get_embeddings(chunks)
            
            if setup_collection() and upload_to_qdrant(chunks, embeddings):
                st.session_state.processed_pdf = True
                st.success("✅ PDF processed and indexed.")
            else:
                st.error("Failed to process and index PDF.")
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        if st.session_state.pdf_path:
            shutil.rmtree(os.path.dirname(st.session_state.pdf_path))
            st.session_state.pdf_path = None

if question and st.session_state.processed_pdf:
    st.info("🔍 Searching for relevant information...")
    query_vector = model.encode([question])[0]
    context_chunks = search_chunks(query_vector)
    context = " ".join(context_chunks)

    st.info("🤖 Generating answer from Gemini...")
    answer = generate_answer_from_gemini(question, context)

    st.subheader("🧠 Answer:")
    st.write(answer)

# Cleanup when the app is closed
if st.session_state.pdf_path and os.path.exists(st.session_state.pdf_path):
    shutil.rmtree(os.path.dirname(st.session_state.pdf_path))
    st.session_state.pdf_path = None
