# QandA RAG

This project implements a Question and Answer (Q&A) system using a Retrieval Augmented Generation (RAG) approach. It allows users to upload PDF documents, which are then processed to extract text, create vector embeddings, and store them for efficient retrieval. Users can then ask questions related to the uploaded documents, and the system will provide answers based on the retrieved information.

## Features
- **PDF Document Upload:** Easily upload PDF files to populate the knowledge base.
- **Text Extraction & Embedding:** Automatically extracts text from PDFs and converts it into vector embeddings for semantic search.
- **Vector Database Integration:** Stores document embeddings in a vector database (e.g., Qdrant) for fast and relevant retrieval.
- **Contextual Question Answering:** Answers user questions by retrieving relevant document chunks and generating concise responses using a Language Model.

## How to Run It

### Prerequisites
- Python 3.8+
- pip

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/QandA_llm.git
   cd QandA_llm/QandA
   ```
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   # On Windows
   .\venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Configuration
- Ensure your `.env` file (or environment variables) are set up with necessary API keys (e.g., OpenAI API key) and vector database connection details (e.g., Qdrant). A `.env.example` might be provided for reference.

### Running the Application
1. Start the application:
   ```bash
   python app.py
   ```
2. Open your web browser and navigate to `http://localhost:5000` (or the port specified in your application).
3. Upload a PDF document and start asking questions!