# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI Knowledge Management System with RAG-based Chatbot built using Streamlit. The system manages the complete knowledge lifecycle: document upload → AI-based knowledge verification and enhancement → RAG-based Q&A chatbot service.

## Development Commands

### Running the Application
```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit development server
streamlit run app.py

# Run with specific port
streamlit run app.py --server.port 8501
```

### Development Setup
```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Architecture Overview

### Single-File Streamlit Architecture
The application follows a simple single-file Streamlit architecture in `app.py` with the following key components:

- **Page Navigation**: Sidebar-based navigation between three main pages (챗봇, 지식등록, 게시판)
- **Session State Management**: Streamlit session state for chat history, board posts, and vector database simulation
- **Modular Page Functions**: Separate functions for each page (chatbot_page, knowledge_registration_page, board_page)

### Core Components

1. **Chatbot Interface** (`chatbot_page()`)
   - Chat interface using `st.chat_message()` and `st.chat_input()`
   - Chat history stored in `st.session_state.chat_history`
   - RAG simulation based on vector database contents

2. **Knowledge Registration** (`knowledge_registration_page()`)
   - File upload supporting TXT, PDF, DOCX, PPTX
   - Document analysis simulation (`analyze_document()`)
   - Two-step process: VectorDB storage and board posting

3. **Knowledge Board** (`board_page()`)
   - Display of enhanced documents with metadata
   - View count tracking and quality scores
   - Expandable post interface

### Key Functions

- `sidebar_navigation()`: Manages page routing and navigation
- `initialize_session_state()`: Sets up session state variables
- `analyze_document()`: Simulates AI analysis and content enhancement
- Main page functions: Handle UI and business logic for each section

### Session State Structure
```python
st.session_state = {
    'chat_history': [],      # List of chat messages
    'board_posts': [],       # List of enhanced documents
    'vector_db': [],         # Simulated vector database
    'current_analysis': {}   # Current document analysis result
}
```

## Technology Stack

- **Frontend**: Streamlit (Python web framework)
- **Document Processing**: Simulated using pandas, python-docx, PyPDF2, python-pptx
- **AI/ML Frameworks**: OpenAI, LangChain, sentence-transformers (currently simulated)
- **Vector Search**: FAISS (currently simulated)

## Development Notes

### Current State
- This is a prototype/MVP with simulated AI functionality
- Real AI integration points are marked but not implemented
- File processing is currently limited to text files with simulation for others

### Future Integration Points
- Replace `analyze_document()` simulation with real AI analysis
- Implement actual vector embedding and storage
- Add real RAG retrieval and generation
- Integrate Azure AI Services as mentioned in README

### File Upload Handling
- Only TXT files are actually processed; others show simulation content
- Upload handling uses Streamlit's native file uploader
- Temporary file processing ready for real document parsing

### State Management
- All data is stored in Streamlit session state (not persistent)
- For production, replace with proper database storage
- Vector database simulation ready for FAISS integration