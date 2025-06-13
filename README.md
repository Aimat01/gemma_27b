# 🔍 RAG with Gemma 3 & Chroma — PDF Question Answering System

This project demonstrates a **Retrieval-Augmented Generation (RAG)** pipeline using:

- 💡 **Google's Gemma 3 27B IT model** (runs locally, no internet required)  
- 🧠 **ChromaDB** as the vector store for document retrieval  
- ✨ **HuggingFace sentence-transformers** for embedding generation  
- 📄 A sample **PDF document** (e.g., Mercedes-AMG GT specifications)  

---

## 📁 Project Structure

```
research_1/
├── gemma-3-27b-it/                 # Local LLM model directory
├── MY20-AMG_GT_Group_WebPDF.pdf   # Example PDF document
├── rag_db/                         # Persisted vector database (Chroma)
├── rag_gemma.py                   # Main script to run the RAG system
└── README.md
```

---

## ⚙️ Installation

### 1. Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate       # On macOS/Linux
.venv\Scripts\activate        # On Windows
```

### 2. Install dependencies

```bash
pip install torch transformers langchain chromadb sentence-transformers
```

> 💡 **Note:** Make sure your GPU supports `bfloat16`. Otherwise, change `torch_dtype=torch.bfloat16` to `torch.float32`.

---

## 💾 Model & Document Setup

- Download and place your **Gemma 3 27B IT model** in `./gemma-3-27b-it`  
- Place your **PDF file(s)** (e.g. user manuals, books, specs) in the project root folder

---

## 🧠 How It Works

### 📄 1. Load & Split the PDF

```python
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = PyPDFLoader("MY20-AMG_GT_Group_WebPDF.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = splitter.split_documents(docs)
```

---

### 🧬 2. Generate Embeddings & Build Vector DB

```python
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma.from_documents(texts, embedding, persist_directory="./rag_db")
retriever = vectordb.as_retriever()
```

> This stores embeddings in `rag_db/` for fast retrieval.

---

### 🤖 3. Load Gemma 3 Model Locally

```python
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
import torch

model = Gemma3ForConditionalGeneration.from_pretrained(
    "./gemma-3-27b-it", torch_dtype=torch.bfloat16, device_map="auto"
).eval()
tokenizer = AutoTokenizer.from_pretrained("./gemma-3-27b-it")
```

---

### 💬 4. Ask a Question with RAG

```python
query = "What engine does the AMG GT have?"

docs = retriever.get_relevant_documents(query)
context = "\n".join([doc.page_content for doc in docs[:3]])

prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=200)

answer = tokenizer.decode(output[0], skip_special_tokens=True)
print(answer)
```

---

## 🧪 Running the Script

Create a Python file `rag_gemma.py`, paste the full pipeline, then run:

```bash
python rag_gemma.py
```

---

## 🛠️ Tips

- You can add more PDFs and rebuild the vector DB  
- Use `.persist()` on `vectordb` to reuse vectors  
- Consider streaming output for longer answers

---

## 🙋‍♂️ Author

Made with ❤️ by **Aimat Kulmakhan**  
3rd Year CS Student @ SDU, Kazakhstan  
Data Science Intern — Freedom Holding

---

## 📜 License

This project is for educational purposes only.  
Model rights belong to Google and Hugging Face.