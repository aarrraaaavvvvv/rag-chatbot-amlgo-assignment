# eBay User Agreement RAG Chatbot

A conversational AI chatbot that answers questions about eBay's User Agreement using a Retrieval-Augmented Generation (RAG) pipeline. Built as part of the Amlgo Labs Junior AI Engineer technical assessment.

---

##  Project Architecture
```
User Query
    │
    ▼
[Query Embedding]  ←  all-MiniLM-L6-v2
    │
    ▼
[FAISS Vector Search]  ←  89 indexed chunks from eBay User Agreement
    │
    ▼
[Top 5 Relevant Chunks Retrieved]
    │
    ▼
[Prompt Construction]  ←  chunks + user query injected into prompt template
    │
    ▼
[LLaMA 3.3 70B via Groq API]  ←  streaming response generation
    │
    ▼
[Streamlit Chat Interface]  ←  real-time token streaming with source display
```

---

## 📁 Folder Structure
```
rag-chatbot/
│
├── data/                          
│   └── training_document.pdf      # source document
│
├── chunks/                        
│   └── chunks.json                # preprocessed text segments
│
├── vectordb/                      
│   └── faiss_index.index          # saved FAISS index file from google colab
│
├── notebooks/                     
│   └── preprocessing.ipynb        # preprocessing colab notebook
│
├── src/                           # core pipeline modules
│   ├── retriever.py               # semantic search using FAISS
│   ├── generator.py               # LLM response generation via Groq
│   └── pipeline.py                # connects retriever and generator
│
├── app.py                         # Streamlit chatbot interface
├── preprocess.py                  # one-time preprocessing script
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚙️ Tech Stack

| Component | Choice | Reason |
|---|---|---|
| Embedding Model | all-MiniLM-L6-v2 | Lightweight, fast, strong semantic performance |
| Vector Database | FAISS | Simple, no server needed, saves to a single file |
| LLM | LLaMA 3.3 70B (via Groq) | Powerful, free API, extremely fast inference |
| Frontend | Streamlit | Quick to build, supports streaming |

---

## 🚀 Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/aarrraaaavvvvv/rag-chatbot.git
cd rag-chatbot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables
Create a `.env` file in the root directory:
```
GROQ_API_KEY = your_groq_api_key
```
You can get a free Groq API key at https://console.groq.com

### 4. Preprocessing (already done)
The preprocessing script was used to generate the chunk embeddings and
FAISS index. The output files (`chunks/chunks.json` and
`vectordb/faiss_index.index`) are already included in the repository,
so you can skip directly to running the app.

If you'd like to rerun preprocessing from scratch:
```bash
python preprocess.py
```
Expected output:
```
Extracting text from PDF...
Total characters extracted: 68253
Chunking...
Total chunks: 89
Generating embeddings...
FAISS index saved. Total vectors: 89
```

### 5. Run the chatbot
```bash
streamlit run app.py
```
The app will open automatically at `http://localhost:8501`

---

## 🔍 How the RAG Pipeline Works

### Step 1 — Document Preprocessing
The eBay User Agreement PDF is loaded using `pypdf` and converted to
plain text. The text is split into overlapping chunks of ~1000 characters
with 90 character overlap. The overlap ensures that sentences spanning
chunk boundaries are not lost.

### Step 2 — Embedding Generation
Each chunk is converted into a 384-dimensional vector using
`all-MiniLM-L6-v2` from the sentence-transformers library. These vectors
capture the semantic meaning of each chunk.

### Step 3 — Vector Storage
All 89 chunk embeddings are stored in a FAISS index using L2 distance.
The index is saved to disk as `faiss_index.index` and loaded once at
app startup.

### Step 4 — Retrieval
When a user submits a query, it is embedded using the same MiniLM model.
FAISS performs a nearest-neighbor search and returns the top 5 most
semantically similar chunks.

### Step 5 — Generation
The retrieved chunks are injected into a prompt template along with the
user query and sent to LLaMA 3.3 70B via the Groq API. The response is
streamed back token by token to the Streamlit interface.

---

## 💬 Sample Queries & Responses

**Q: What happens if eBay suspends my account?**
If eBay suspends your account, they may limit, restrict, or downgrade your seller account, delay or remove hosted content, remove any special status associated with your account, remove or demote listings, reduce or eliminate discounts, and take technical and/or legal steps to prevent you from using their Services. Additionally, you may be subject to fees and recovery of expenses for policy monitoring and enforcement.

**Q: How does eBay handle disputes between buyers and sellers?**
eBay facilitates the resolution of disputes between buyers and sellers through various programs. However, eBay has no control over and does not guarantee the existence, quality, safety, or legality of items advertised, the truth or accuracy of users' content or listings, or that a buyer or seller will actually complete a transaction or return an item.

**Q: Can eBay change its policies without notice?**
According to the eBay User Agreement, eBay can change its selling fees without advance notice for temporary promotions or changes that result in the reduction of fees. However, for other changes, they will provide 30 days' notice by posting the amended terms on www.eBay.com, and also notify users through the eBay Message Center and/or by email.

---

## ⚠️ Known Limitations

- The chatbot only answers based on the provided eBay User Agreement
document. Questions requiring external knowledge (e.g. exact fee
percentages) may not be answered fully as that information is referenced
as an external link in the document.

- Chunk boundaries occasionally split mid-clause in very long legal
sentences. Increasing chunk size or using a sentence-aware splitter
could improve this.

- The embedding model runs on CPU which adds a small delay on first load.

- Groq API rate limits apply on the free tier.

---

## 🎥 Demo



---

## 📦 Requirements
```
streamlit
faiss-cpu
sentence-transformers
groq
pypdf
numpy
python-dotenv
```
```
