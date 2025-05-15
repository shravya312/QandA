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
import graphviz
import re

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
    
    marks_match = re.search(r'(\d+)\s*marks', query, re.IGNORECASE)
    if marks_match:
        marks = int(marks_match.group(1))
        if marks >= 8:
            extra_instruction = (
                "This is a high-mark question. Write a long, detailed, and well-structured answer. "
                "Include introduction, step-by-step explanation, algorithm, math, advantages, disadvantages, applications, and examples."
            )
        elif marks >= 5:
            extra_instruction = (
                "This is a medium-mark question. Write a moderately detailed answer with explanation, steps, and at least one example."
            )
        else:
            extra_instruction = (
                "This is a short-mark question. Write a concise answer."
            )
    else:
        extra_instruction = ""

    prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Instructions: {extra_instruction}\n\n"
        f"Answer:"
    )
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Error from Gemini: {str(e)}"

def generate_mcqs_from_context(context, num_questions=5):
    """
    Use the LLM to generate MCQs from the context.
    """
    prompt = (
        f"Generate {num_questions} multiple choice questions (MCQs) from the following context. "
        f"For each question, use this format:\n"
        f"Q: <question text>\n"
        f"A) <option 1>\n"
        f"B) <option 2>\n"
        f"C) <option 3>\n"
        f"D) <option 4>\n"
        f"Answer: <A/B/C/D>\n\n"
        f"Context:\n{context}\n"
    )
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"Error generating MCQs: {e}")
        return None

def parse_mcqs(mcq_text):
    """
    Parse the MCQ text into a list of dicts.
    """
    questions = []
    blocks = re.split(r'\n(?=Q: )', mcq_text)
    for block in blocks:
        q = {}
        lines = block.strip().split('\n')
        if len(lines) < 6:
            continue
        q['question'] = lines[0][3:].strip()
        q['options'] = [line[3:].strip() for line in lines[1:5]]
        answer_line = lines[5]
        q['answer'] = answer_line.split(':')[-1].strip()
        questions.append(q)
    return questions

def generate_flowchart_from_context(context, topic):
    """
    Use the LLM to generate a flowchart description in DOT language.
    """
    prompt = (
        f"Given the following context from a textbook or manual, generate a flowchart in Graphviz DOT format "
        f"for the topic: '{topic}'. Only output the DOT code, nothing else.\n\n"
        f"Context:\n{context}\n"
    )
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"Error generating flowchart: {e}")
        return None

# ===== Streamlit UI =====
st.set_page_config(layout="wide")
st.title("📄 PDF Question Answering System with RAG")

# Initialize session state
if 'processed_pdf' not in st.session_state:
    st.session_state.processed_pdf = False
if 'pdf_path' not in st.session_state:
    st.session_state.pdf_path = None
if 'context_chunks' not in st.session_state:
    st.session_state.context_chunks = []
if 'text' not in st.session_state:
    st.session_state.text = None
if 'chunks' not in st.session_state:
    st.session_state.chunks = []

pdf = st.file_uploader("Upload a PDF file", type=["pdf"])

if pdf and not st.session_state.processed_pdf:
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, "uploaded.pdf")
    try:
        with open(temp_path, "wb") as f:
            f.write(pdf.read())
        st.session_state.pdf_path = temp_path
        st.info("⏳ Extracting and processing PDF...")
        text = extract_text_from_pdf(temp_path)
        st.session_state.text = text
        if text:
            chunks = chunk_text(text)
            st.session_state.chunks = chunks
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

# Tabs for Q&A and MCQ only (remove Flowchart Generation tab)
qna_tab, mcq_tab = st.tabs(["PDF Q&A", "MCQ Generation"])

with qna_tab:
    st.header("Ask a Question from the PDF")
    question = st.text_input("Ask a question based on the PDF content:")
    if question and st.session_state.processed_pdf:
        st.info("🔍 Searching for relevant information...")
        query_vector = model.encode([question])[0]
        context_chunks = search_chunks(query_vector)
        st.session_state.context_chunks = context_chunks
        context = " ".join(context_chunks)
        st.info("🤖 Generating answer from Gemini...")
        answer = generate_answer_from_gemini(question, context)
        st.subheader("🧠 Answer:")
        st.write(answer)

with mcq_tab:
    st.header("Generate MCQs from PDF Concepts")
    num_questions = st.slider("Number of MCQs", 1, 10, 5)
    if st.button("Generate MCQs") and st.session_state.processed_pdf:
        context = " ".join(st.session_state.context_chunks) if st.session_state.context_chunks else st.session_state.text
        mcq_text = generate_mcqs_from_context(context, num_questions)
        if mcq_text:
            questions = parse_mcqs(mcq_text)
            for i, q in enumerate(questions):
                st.markdown(f"**Q{i+1}: {q['question']}**")
                st.markdown(f"A) {q['options'][0]}")
                st.markdown(f"B) {q['options'][1]}")
                st.markdown(f"C) {q['options'][2]}")
                st.markdown(f"D) {q['options'][3]}")
                st.markdown('---')
        else:
            st.warning("No MCQs could be generated.")

# Cleanup when the app is closed
if st.session_state.pdf_path and os.path.exists(st.session_state.pdf_path):
    shutil.rmtree(os.path.dirname(st.session_state.pdf_path))
    st.session_state.pdf_path = None
