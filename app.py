import os
import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, Filter, FieldCondition, MatchValue, PayloadSchemaType, CreateFieldIndex
from qdrant_client.models import PointStruct 
from sentence_transformers import SentenceTransformer 
from pypdf import PdfReader
import google.generativeai as genai
from dotenv import load_dotenv
import tempfile
import shutil
import numpy as np
import re
import hashlib
import uuid
from rank_bm25 import BM25Okapi

load_dotenv()


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
MIN_SIMILARITY_SCORE = 0.62  # Only accept chunks at/above this score

# ===== Core Functions =====
def generate_pdf_hash(pdf_file_path):
    """Generate a unique hash for the PDF file based on its content"""
    try:
        with open(pdf_file_path, 'rb') as f:
            file_content = f.read()
        return hashlib.md5(file_content).hexdigest()
    except Exception as e:
        st.error(f"Error generating PDF hash: {str(e)}")
        return None

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

def tokenize_text(text):
    """Tokenize text for BM25 processing."""
    return text.lower().split()

def get_embeddings(chunks):
    """Convert text chunks into embeddings"""
    try:
        return model.encode(chunks)
    except Exception as e:
        st.error(f"Error generating embeddings: {str(e)}")
        return None

def is_valid_embeddings(embeddings):
    """Check if embeddings are valid and not empty"""
    if embeddings is None:
        return False
    if isinstance(embeddings, np.ndarray):
        return embeddings.size > 0
    if hasattr(embeddings, '__len__'):
        return len(embeddings) > 0
    return False

def setup_collection():
    """Create Qdrant collection if it doesn't exist"""
    try:
        # Check if collection exists
        collections = qdrant_client.get_collections()
        collection_exists = any(col.name == COLLECTION_NAME for col in collections.collections)
        
        if not collection_exists:
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE
                )
            )
            st.info(f"Created new collection: {COLLECTION_NAME}")
        else:
            st.info(f"Using existing collection: {COLLECTION_NAME}")
        
        # Create index for pdf_hash field if it doesn't exist
        try:
            qdrant_client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="pdf_hash",
                field_schema=PayloadSchemaType.KEYWORD
            )
        except Exception:
            # Index might already exist, which is fine
            pass
        
        return True
    except Exception as e:
        st.error(f"Error setting up collection: {str(e)}")
        return False

def check_pdf_embeddings_exist(pdf_hash):
    """Check if embeddings for a PDF already exist in Qdrant"""
    try:
        result = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="pdf_hash",
                        match=MatchValue(value=pdf_hash)
                    )
                ]
            ),
            limit=1
        )
        return len(result[0]) > 0
    except Exception as e:
        st.warning(f"Error checking existing embeddings: {str(e)}")
        return False

def get_existing_embeddings(pdf_hash):
    """Retrieve existing chunks for a PDF from Qdrant (no vectors needed)."""
    try:
        result = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="pdf_hash",
                        match=MatchValue(value=pdf_hash)
                    )
                ]
            ),
            with_vectors=False,
            limit=10000  # Large limit to get all chunks
        )
        
        if result[0]:
            # Sort by id to maintain chunk order
            sorted_points = sorted(result[0], key=lambda x: x.id)
            chunks = [point.payload.get("text", "") for point in sorted_points]
            return chunks, None
        return [], None
    except Exception as e:
        st.error(f"Error retrieving existing embeddings: {str(e)}")
        return [], None

from qdrant_client.http import models

def clear_all_embeddings():
    try:
        qdrant_client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=[])  # empty filter ⇒ match all points
            )
        )
        print("🗑️ Cleared all embeddings and payloads from collection")
    except Exception as e:
        print(f"Error clearing embeddings: {e}")

def load_all_chunks_for_bm25():
    """Loads all chunks from Qdrant to build a global BM25 model."""
    try:
        scroll_result, _ = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100000,  # Adjust limit as needed for total number of chunks
            with_payload=True,
            with_vectors=False
        )
        all_chunks = [point.payload.get("text", "") for point in scroll_result if point.payload.get("text")]
        if all_chunks:
            tokenized_all_chunks = [tokenize_text(chunk) for chunk in all_chunks]
            st.session_state.bm25_model = BM25Okapi(tokenized_all_chunks)
            st.session_state.all_chunks = all_chunks # Store original chunks for retrieval
            st.info("BM25 model initialized with all existing chunks.")
        else:
            st.info("No chunks found in Qdrant for BM25 initialization.")
    except Exception as e:
        st.error(f"Error loading all chunks for BM25: {str(e)}")

def recreate_collection_with_schema():
    """Recreate collection with proper payload schema"""
    try:
        # Delete existing collection
        try:
            qdrant_client.delete_collection(COLLECTION_NAME)
        except:
            pass  # Collection might not exist
        
        # Create new collection
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )
        
        # Create index for pdf_hash field
        qdrant_client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="pdf_hash",
            field_schema=PayloadSchemaType.KEYWORD
        )

        # Validate index works by performing a filtered scroll (limit 1)
        try:
            qdrant_client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="pdf_hash",
                            match=MatchValue(value="_validation_check_")
                        )
                    ]
                ),
                with_vectors=False,
                limit=1
            )
        except Exception as validation_error:
            st.error(f"Validation after recreation failed. The index may not be active yet: {validation_error}")
            return False

        st.success("Collection recreated and pdf_hash index validated.")
        load_all_chunks_for_bm25()
        return True
    except Exception as e:
        st.error(f"Error recreating collection: {str(e)}")
        return False

def upload_to_qdrant(chunks, embeddings, pdf_hash):
    """Upload data to Qdrant with PDF hash for identification"""
    if not chunks or not is_valid_embeddings(embeddings):
        return False
    
    try:
        # Ensure vectors are plain Python lists of floats and provide stable string ids
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vector_list = embedding.tolist() if isinstance(embedding, np.ndarray) else list(embedding)
            point_id = uuid.uuid5(uuid.NAMESPACE_DNS, f"{pdf_hash}-{i}")
            points.append(
                PointStruct(
                    id=str(point_id),
                    vector=vector_list,
                    payload={"text": chunk, "pdf_hash": pdf_hash}
                )
            )
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        return True
    except Exception as e:
        st.error(f"Error uploading to Qdrant: {str(e)}")
        return False

def search_chunks(query_text):
    """Retrieve most relevant chunks from all available PDFs using hybrid search."""
    try:
        # --- Dense Retrieval (Qdrant) ---
        query_vector = model.encode([query_text])[0]
        qdrant_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=20, # Get more candidates for re-ranking
        )
        dense_hits = {hit.payload.get("text", ""): hit.score for hit in qdrant_results}

        # --- Sparse Retrieval (BM25) ---
        if 'bm25_model' in st.session_state and 'all_chunks' in st.session_state:
            bm25_model = st.session_state.bm25_model
            all_chunks = st.session_state.all_chunks
            tokenized_query = tokenize_text(query_text)
            bm25_scores = bm25_model.get_scores(tokenized_query)
            
            # Pair scores with original chunks
            sparse_hits = {}
            for i, score in enumerate(bm25_scores):
                original_chunk = all_chunks[i]
                if score > 0: # Only consider chunks with a positive BM25 score
                    sparse_hits[original_chunk] = score
        else:
            sparse_hits = {}

        # --- Hybrid Scoring ---
        combined_scores = {}
        alpha = 0.5 # Weight for dense vs sparse

        all_chunks = set(dense_hits.keys()).union(set(sparse_hits.keys()))

        if not all_chunks: # If no chunks from either, return empty
            return []
        
        # Normalize scores (simple min-max for now, can be improved)
        max_dense_score = max(dense_hits.values()) if dense_hits else 1.0
        max_sparse_score = max(sparse_hits.values()) if sparse_hits else 1.0

        for chunk in all_chunks:
            dense_score = dense_hits.get(chunk, 0.0) / max_dense_score
            sparse_score = sparse_hits.get(chunk, 0.0) / max_sparse_score
            combined_scores[chunk] = (alpha * dense_score) + ((1 - alpha) * sparse_score)
        
        # Sort and return top chunks
        sorted_chunks = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)
        return [chunk for chunk, score in sorted_chunks[:20]]

    except Exception as e:
        st.error(f"Error performing hybrid search: {str(e)}")
        return []

def re_rank_chunks(query_text, candidate_chunks, top_k=5):
    """Re-ranks candidate chunks based on semantic similarity to the query."""
    if not candidate_chunks:
        return []

    # Encode query and candidate chunks
    query_embedding = model.encode([query_text])[0]
    chunk_embeddings = model.encode(candidate_chunks)

    # Calculate cosine similarity between query and each chunk
    # Using numpy's dot product for cosine similarity after normalization in re_rank_chunks
    query_embedding_norm = np.linalg.norm(query_embedding)
    chunk_embeddings_norm = np.linalg.norm(chunk_embeddings, axis=1)

    # Avoid division by zero for zero vectors
    if query_embedding_norm == 0:
        return []
    chunk_embeddings_norm[chunk_embeddings_norm == 0] = 1e-12 # Small epsilon to avoid div by zero

    similarities = np.dot(chunk_embeddings, query_embedding) / (chunk_embeddings_norm * query_embedding_norm)

    # Pair chunks with their similarity scores
    scored_chunks = list(zip(candidate_chunks, similarities))

    # Sort by similarity score in descending order
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    # Return top_k re-ranked chunks
    return [chunk for chunk, score in scored_chunks[:top_k]]

def expand_query(query, num_terms=3):
    """Expands the user query with additional terms using Gemini."""
    prompt = (
        f"Generate {num_terms} related terms or short phrases for the following query. "
        f"Output only the terms, separated by commas, no other text.\n"
        f"Query: {query}\n"
        f"Related terms:"
    )
    try:
        response = gemini_model.generate_content(prompt)
        expanded_terms = [term.strip() for term in response.text.strip().split(',') if term.strip()]
        return " ".join(expanded_terms) # Return as a space-separated string
    except Exception as e:
        st.warning(f"Error expanding query: {e}")
        return ""

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

# Sidebar for additional options
with st.sidebar:
    st.header("🔧 Options")
    if st.button("🗑 Clear All Embeddings", help="This will delete all stored embeddings from Qdrant"):
        if clear_all_embeddings():
            st.session_state.processed_pdf = False
            st.session_state.pdf_path = None
            st.session_state.context_chunks = []
            st.session_state.text = None
            st.session_state.chunks = []
            st.rerun()
    
    if st.button("🔄 Recreate Collection", help="Recreate collection with proper schema (fixes index errors)"):
        if recreate_collection_with_schema():
            st.session_state.processed_pdf = False
            st.session_state.pdf_path = None
            st.session_state.context_chunks = []
            st.session_state.text = None
            st.session_state.chunks = []
            st.rerun()
    
    st.header("ℹ Information")
    st.info("""
    *How it works:*
    - Each PDF gets a unique hash based on its content
    - Embeddings are stored with this hash as a key
    - If you upload the same PDF again, existing embeddings are reused
    - This saves time and computational resources
    """)

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

# Initialize Qdrant and BM25 on startup
if setup_collection():
    load_all_chunks_for_bm25()

pdf = st.file_uploader("Upload a PDF file", type=["pdf"])

# Detect if a new PDF is uploaded and reset session state
if pdf is not None:
    if 'last_uploaded_filename' not in st.session_state or st.session_state.last_uploaded_filename != pdf.name:
        st.session_state.processed_pdf = False
        st.session_state.pdf_path = None
        st.session_state.context_chunks = []
        st.session_state.text = None
        st.session_state.chunks = []
        st.session_state.mcq_questions = []
        st.session_state.user_mcq_answers = []
        st.session_state.mcq_submitted = False
        st.session_state.last_uploaded_filename = pdf.name

if pdf and not st.session_state.processed_pdf:
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, "uploaded.pdf")
    try:
        with open(temp_path, "wb") as f:
            f.write(pdf.read())
        st.session_state.pdf_path = temp_path
        
        # Generate PDF hash for identification
        pdf_hash = generate_pdf_hash(temp_path)
        if not pdf_hash:
            st.error("Failed to generate PDF hash.")
        else:
            # Store pdf hash for checking existing embeddings
            st.session_state.pdf_hash = pdf_hash
            st.info("⏳ Checking for existing embeddings...")
            
            # Check if embeddings already exist
            should_upload = True
            if check_pdf_embeddings_exist(pdf_hash):
                st.info("📚 Found existing embeddings for this PDF. Loading from Qdrant...")
                chunks, embeddings = get_existing_embeddings(pdf_hash)
                if chunks:
                    st.session_state.chunks = chunks
                    st.session_state.processed_pdf = True
                    st.success("✅ PDF embeddings loaded from existing data.")
                    # We already have data in Qdrant, no need to upload again
                    should_upload = False
                else:
                    st.warning("Existing embeddings found but couldn't retrieve them. Processing PDF again...")
                    # Fall through to normal processing
                    chunks, embeddings = None, None
            else:
                chunks, embeddings = None, None
            
            # If no existing embeddings, process the PDF
            if not chunks or not is_valid_embeddings(embeddings):
                st.info("⏳ Extracting and processing PDF...")
                text = extract_text_from_pdf(temp_path)
                st.session_state.text = text
                if text:
                    chunks = chunk_text(text)
                    st.session_state.chunks = chunks
                    tokenized_chunks = [tokenize_text(chunk) for chunk in chunks]
                    bm25_model = BM25Okapi(tokenized_chunks)
                    st.session_state.bm25_model = bm25_model
                    st.session_state.tokenized_chunks = tokenized_chunks
                    embeddings = get_embeddings(chunks)
            
            # Upload to Qdrant only if we created new embeddings in this run
            if should_upload and chunks and is_valid_embeddings(embeddings) and setup_collection():
                if upload_to_qdrant(chunks, embeddings, pdf_hash):
                    st.session_state.processed_pdf = True
                    st.success("✅ PDF processed and indexed.")
                    load_all_chunks_for_bm25() # Reload BM25 with new chunks
                else:
                    st.error("Failed to upload embeddings to Qdrant.")
            elif not st.session_state.processed_pdf and (not chunks or not is_valid_embeddings(embeddings)):
                st.error("Failed to process PDF.")
            
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
        st.info("✨ Expanding query...")
        expanded_terms = expand_query(question)
        expanded_query_text = f"{question} {expanded_terms}".strip()
        
        st.info("🔍 Searching for relevant information...")
        # query_vector = model.encode([question])[0] # No longer needed, search_chunks takes text
        
        # Perform hybrid search to get candidate chunks
        candidate_chunks = search_chunks(expanded_query_text)
        
        # Re-rank the candidate chunks
        re_ranked_chunks = re_rank_chunks(expanded_query_text, candidate_chunks, top_k=5)
        
        st.session_state.context_chunks = re_ranked_chunks
        context = " ".join(re_ranked_chunks)
        st.info("🤖 Generating answer from Gemini...")
        answer = generate_answer_from_gemini(question, context)
        st.subheader("🧠 Answer:")
        st.write(answer)

with mcq_tab:
    st.header("Generate MCQs from PDF Concepts")
    num_questions = st.slider("Number of MCQs", 1, 10, 5)

    # Button to generate MCQs
    if st.button("Generate MCQs") and st.session_state.processed_pdf:
        context = " ".join(st.session_state.context_chunks) if st.session_state.context_chunks else st.session_state.text
        mcq_text = generate_mcqs_from_context(context, num_questions)
        if mcq_text:
            questions = parse_mcqs(mcq_text)
            if questions:
                st.session_state.mcq_questions = questions
                st.session_state.user_mcq_answers = [None] * len(questions)
                st.session_state.mcq_submitted = False
            else:
                st.warning("No MCQs could be parsed.")
        else:
            st.warning("No MCQs could be generated.")

    # Only display MCQs if they exist in session state
    if 'mcq_questions' in st.session_state and st.session_state.mcq_questions:
        questions = st.session_state.mcq_questions
        for i, q in enumerate(questions):
            st.markdown(f"Q{i+1}: {q['question']}")
            st.session_state.user_mcq_answers[i] = st.radio(
                "Select your answer:",
                options=['A', 'B', 'C', 'D'],
                key=f"mcq_{i}"
            )
            st.markdown(f"A) {q['options'][0]}")
            st.markdown(f"B) {q['options'][1]}")
            st.markdown(f"C) {q['options'][2]}")
            st.markdown(f"D) {q['options'][3]}")
            st.markdown('---')

        # Submit button
        if st.button("Submit Quiz"):
            st.session_state.mcq_submitted = True

        # Show score and feedback only after submission
        if st.session_state.get('mcq_submitted', False):
            score = 0
            questions = st.session_state.mcq_questions
            user_answers = st.session_state.user_mcq_answers
            for i, q in enumerate(questions):
                is_correct = user_answers[i] == q['answer']
                st.markdown(f"Q{i+1}: {q['question']}")
                for idx, opt in enumerate(['A', 'B', 'C', 'D']):
                    option_text = f"{opt}) {q['options'][idx]}"
                    if user_answers[i] == opt:
                        if is_correct:
                            st.markdown(f"<span style='color: green; font-weight: bold'>Your answer: {option_text} ✔</span>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<span style='color: red; font-weight: bold'>Your answer: {option_text} ❌</span>", unsafe_allow_html=True)
                    elif not is_correct and q['answer'] == opt:
                        st.markdown(f"<span style='color: green; font-weight: bold'>Correct answer: {option_text} ✔</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"{option_text}")
                st.markdown('---')
                if is_correct:
                    score += 1
            st.success(f"Your score: {score}/{len(questions)}")

# Cleanup when the app is closed
if st.session_state.pdf_path and os.path.exists(st.session_state.pdf_path):
    shutil.rmtree(os.path.dirname(st.session_state.pdf_path))
    st.session_state.pdf_path = None