🤖 Multi-Agent AI Chatbot with RAG & Retrieval
==============================================

**Final Project Submission** **Developer:** \[Md.Sayed Bin Anwar Akib\]

**Submission Date:** 16 March 2026

This project is a sophisticated chatbot application built with **LangChain**, **Streamlit**, and **LangSmith**. It features a multi-agent architecture capable of real-time web grounding, document-based reasoning (RAG), and Optical Character Recognition (OCR).

🚀 Key Features
---------------

*   **Multi-Agent System:** Dynamically switch between a **Search Agent** for live web data and a **RAG Agent** for private document analysis.
    
*   **RAG Pipeline:** Utilizes **FAISS** for high-performance vector similarity search.
    
*   **OCR Integration:** Extract and query text from images and scanned PDFs for seamless information retrieval.
    
*   **Observability:** Full integration with **LangSmith** for deep tracing of agent executions, tool calls, and LLM chains.
    
*   **Streaming UI:** A modern, responsive chat interface built with **Streamlit** featuring real-time token streaming.
    

🛠️ Technical Stack
-------------------

**ComponentTechnologyRoleLLM**GPT-OSS-120B (via Groq)Core reasoning engine**Orchestration**LangChain / LangGraphAgent logic and tool management**Tracing**LangSmithProduction monitoring and debugging**Vector DB**FAISSSemantic storage for RAG**Embeddings**all-MiniLM-L6-v2Local high-speed vector generation**UI**StreamlitFrontend interface

📂 Project Structure
--------------------

├── agents/
│   ├── search_agent.py   # Internet Search Tool & Grounding logic
│   ├── rag_agent.py      # Vector DB logic & RAG pipeline
│   └── ocr_tool.py       # Image processing & OCR logic
├── utils/
│   ├── llm_config.py     # Groq & LangSmith initialization
│   └── document_loader.py # Logic for parsing PDFs and TXT files
├── data/                 # Local storage for indexed documents
├── app.py                # Main Streamlit application entry point
├── requirements.txt      # Project dependencies
└── .env                  # API keys (Groq, LangSmith)

⚙️ Setup & Installation
-----------------------

### 1\. Clone & Environment

Bash

git clone [your-repo-link-here]
cd simple_chatbot
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

### 2\. Configuration

Create a **.env** file in the root directory and add your credentials:

Code snippet

GROQ_API_KEY=your_groq_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key_here
LANGCHAIN_PROJECT=Final_Project_Chatbot

### 3\. Run the Application

Bash

streamlit run app.py

🔍 Agent Functionality
----------------------

### 🌐 Search Agent

Uses a dedicated search tool to perform real-time grounding. It identifies when a user query requires up-to-date information (e.g., "What are the latest AI news trends?") and provides cited sources.

### 📚 RAG Agent (Retrieval-Augmented Generation)

*   **Ingestion:** Processes PDFs/Text files into manageable 500-character chunks.
    
*   **Indexing:** Generates embeddings locally and stores them in a **FAISS** index.
    
*   **Retrieval:** Performs similarity search to find the most relevant context to ground the LLM's response.
    

### 🖼️ OCR Tooling

Integrated within the RAG agent, this tool allows the user to upload images or scanned documents. The system extracts text using OCR before indexing it into the vector store, allowing you to "chat" with images.

📊 Monitoring (LangSmith)
-------------------------

Every agent execution, tool call, and retrieval step is traced. To view the project traces:

1.  Log in to your [LangSmith Dashboard](https://smith.langchain.com/).
    
2.  Select the Final\_Project\_Chatbot project.
    
3.  Inspect **Latency**, **Token Usage**, and the **Chain of Thought** for every user interaction.
    

🎥 Presentation & Demo
----------------------

*   **YouTube Demo:** \[Link to your YouTube Video\]
    
*   **GitHub Repository:** \[Link to this Repo\]