<h1>AI-Based Legal ChatBot Using Retrieval Augmented Generation (RAG)</h1>
<p>An AI-powered Legal Assistant designed to make Indian laws easy to understand for common people, students, and researchers.
The system uses Retrieval-Augmented Generation (RAG) to provide accurate, contextual, and explainable legal information based on authenticated legal documents.</p>
<p><b>Disclaimer: This project provides legal information for educational purposes only and does not replace professional legal advice.</b></p>

<h1>🎯 Project Objective</h1>
<ul>
  <li>Bridge the gap between complex legal language and common users</li>
  <li>Provide accurate, up-to-date Indian legal information</li>
  <li>Support multiple Indian languages</li>
  <li>
    Enable document-based legal reasoning using AI
  </li>
</ul>

<h1>Key Features</h1>
<ul>
  <h3>User Features</h3>
  <li>🤖 AI-based legal question answering</li>
  <li>📚 Covers major Indian laws: (IPC, BNS, CONSUMER PROTECT ACT, IT ACT 2000)</li>
  <li>🌐 Multilingual support:(Hindi , English, Kannada, Tamil)</li>
  <li>📄 Chat transcript download (TXT & PDF)</li>
  <li>🕘 Conversation history management</li>
  <li>📰 Latest legal news & updates</li>
</ul>
<ul>
  <h3>🔹Admin Features</h3>
  <li>📂 Upload legal documents (PDF)</li>
  <li>🏷️ Act-wise document tagging (act_name)</li>
  <li>🧩 Automatic document chunking & vector storage</li>
  <li>📊 System statistics dashboard</li>
  <li>🗑️ Delete individual documents</li>
  <li>❌ Clear entire vector database</li>
  <li>📜 System logs & backend health monitoring</li>
</ul>

<h1>🏗️ System Architecture (High-Level)</h1>
<p>
  User Interface (HTML + Tailwind)
        ↓
  FastAPI Backend (REST APIs)
        ↓
  RAG Engine (LangChain)
        ↓
  ChromaDB Vector Database
        ↓
  LLM (NIVIDIA)
</p>

<h1>⚙️ Technologies Used</h1>
<ul>
  <h3>Frontend</h3>
  <li>
    HTML5
  </li>
  <li>Tailwind CSS</li>
  <li>JavaScript</li>
</ul>
<ul>
  <h3>Backend</h3>
  <li>FastAPI</li>
  <li>Python 3.12</li>
</ul>
<ul>
  <h3>AI</h3>
  <li>LangChain</li>
  <li>Retrieval-Augmented Generation (RAG)</li>
  <li>Vector Embeddings</li>
  <li>Large Language Models (LLMs)</li>
</ul>

<ul>
  <h3>Database</h3>
  <ul>ChromaDB (Vector DB)</ul>
</ul>

<h1>🛠️ Installation & Setup</h1>
<h3>
  1️⃣ Clone the Repository
</h3>
<p>
  git clone https://github.com/your-username/legal-ai-chatbot.git
  cd legal-ai-chatbot
</p>
<h3>
  Create Virtual Environment
</h3>
<p>
  python -m venv venv
  source venv/Scripts/activate   # Windows
</p>
<h3>
  Install Dependencies
</h3>
<p>
  pip install -r requirements.txt
</p>
<h3>
  Run the Backend Server
</h3>
<p>
  uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8001
</p>
<h3>
  Open Frontend
</h3>
<p>
  Open index.html in your browser.
</p>

<h1>🧪 How the RAG System Works</h1>
<ol>
  <li>Legal documents are uploaded (PDF)</li>
  <li>
    Documents are split into chunks
  </li>
  <li>
    Chunks are converted into vector embeddings
  </li>
  <li>
    Stored in ChromaDB
  </li>
  <li>
    User query is embedded
  </li>
  <li>
    Relevant chunks are retrieved
  </li>
  <li>
    LLM generates an answer using retrieved context
  </li>
</ol>


