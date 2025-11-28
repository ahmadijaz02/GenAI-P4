# 🩺 Medical RAG QA System

A Retrieval-Augmented Generation (RAG) system using LangChain and Google's Gemini API to answer medical questions based on clinical transcriptions.

## 📋 Project Structure

```
GenAI-P3/
├── preprocess_data.py       # Data preprocessing and vector store creation
├── rag_pipeline.py          # RAG pipeline with retriever and LLM
├── app.py                   # Streamlit web application
├── Task1/
│   ├── mtsamples.csv        # Medical transcription dataset
│   └── requirements.txt      # Python dependencies
└── vectorstore/             # FAISS vector store (created by preprocessing)
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd Task1
pip install -r requirements.txt
cd ..
```

### 2. Preprocess Data (One-time setup)

This creates a FAISS vectorstore from the medical dataset:

```bash
python preprocess_data.py
```

**Note:** This takes 10-15 minutes on first run as it:
- Loads 5030 medical transcriptions
- Chunks them into ~8000+ segments
- Generates embeddings using HuggingFace
- Builds FAISS index
- Saves to `vectorstore/` for fast loading

The vectorstore is saved so the app loads instantly on subsequent runs.

### 3. Run the Streamlit App

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501` and start asking medical questions!

### 4. Evaluate the System

The app includes an evaluation section where you can:
- Select a specific medical question from a list
- Run the RAG system on that question
- View the answer and retrieved sources

## 📊 System Architecture

### Data Preprocessing
1. Load medical transcriptions from CSV
2. Create LangChain documents with metadata
3. Split into 2000-token chunks with 200-token overlap
4. Generate embeddings using HuggingFace sentence transformers
5. Build FAISS vector index and save to disk

### RAG Pipeline
1. **Retriever**: Search FAISS index for top-5 relevant documents
2. **Context**: Format retrieved documents as context
3. **Prompt**: Combine context with user question
4. **LLM**: Use Google Gemini to generate answer
5. **Output**: Return answer with source documents

### Web Interface (Streamlit)
- **QA Section**: Ask medical questions and get instant answers
- **Source Display**: View retrieved documents with sources
- **Evaluation Section**: Test system on predefined medical queries

## 🔑 API Key Setup

The system uses a hardcoded API key in `rag_pipeline.py`. For production, replace with your own Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

## 📈 Supported Questions

The evaluation section includes 33 predefined medical questions covering:
- Allergies and Immunology
- Bariatric Surgery
- Cardiovascular Conditions
- Nephrology and Kidney Disease
- Neurology and Pain
- Orthopedics
- Gastroenterology
- Pulmonology and Respiratory
- Endocrinology and Metabolic
- General Medical Examination

## 🛠️ Customization

### Add More Evaluation Questions
Edit the `questions` list in the Evaluation section of `app.py` to add custom queries.

### Adjust Retrieval Parameters
In `rag_pipeline.py`, modify the retriever settings:

```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
```
Change `k` to retrieve more or fewer documents.

### Change LLM Temperature
In `rag_pipeline.py`, adjust the temperature parameter:

```python
llm = ChatGoogleGenerativeAI(model=MY_GOOGLE_MODEL, temperature=0.3)
```
Lower values = more factual, Higher values = more creative.

## 📚 Deliverables

- ✅ **Data Preprocessing**: `preprocess_data.py` converts CSV to FAISS vector store
- ✅ **RAG Pipeline**: `rag_pipeline.py` implements retriever + Gemini LLM chain
- ✅ **Web Application**: `app.py` provides QA and evaluation interface
- ✅ **Vector Store**: FAISS index for fast document retrieval
- ✅ **Medical Dataset**: 5030 clinical transcriptions with metadata

## 🐛 Troubleshooting

### Vectorstore not found
Run preprocessing to create the vector store:
```bash
python preprocess_data.py
```

### API quota exceeded
Ensure you have a valid Google API key with billing enabled. Only LLM calls use quota (embeddings are local).

### Missing dependencies
Reinstall all packages:
```bash
cd Task1
pip install -r requirements.txt
cd ..
```

## 📖 References

- [Medical Transcriptions Dataset](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions)
- [LangChain Documentation](https://python.langchain.com/)
- [Google Gemini API](https://ai.google.dev/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [HuggingFace Sentence Transformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

---

**Created**: November 2025  
**Medical RAG QA System**
