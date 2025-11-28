import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
try:
    MY_GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except (FileNotFoundError, KeyError):
    MY_GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

MY_GOOGLE_MODEL = os.environ.get("GOOGLE_MODEL", "models/gemini-2.0-flash")

def load_vectorstore(vectorstore_path="vectorstore"):
    if not os.path.exists(vectorstore_path):
        raise FileNotFoundError(f"Vectorstore not found at {vectorstore_path}")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)

def setup_rag_pipeline(vectorstore, api_key=None):
    if api_key is None:
        api_key = MY_GOOGLE_API_KEY
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = ChatGoogleGenerativeAI(model=MY_GOOGLE_MODEL, temperature=0.3, google_api_key=api_key)
    
    prompt_template = """You are a medical assistant. Use the context below to answer the question.
If the answer is supported by the context, provide a clear answer. If context is incomplete, provide
a best-effort response and mark uncertainty. Never refuse; always attempt to answer using provided context.

Context:
{context}

Question: {question}

Answer:"""
    
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    class RAGChain:
        def __init__(self, retriever, llm, prompt):
            self.retriever = retriever
            self.llm = llm
            self.prompt = prompt
        
        def invoke(self, inputs):
            query = inputs.get("query") or inputs.get("question")
            source_docs = self.retriever.invoke(query)
            context_pieces = [f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}" for doc in source_docs]
            context = "\n\n".join(context_pieces)
            formatted_prompt = self.prompt.format(context=context, question=query)
            answer = self.llm.invoke(formatted_prompt)
            text_answer = answer.content if hasattr(answer, 'content') else str(answer)
            sources = [doc.metadata.get("source", "unknown") for doc in source_docs]
            return {"result": text_answer, "source_documents": source_docs, "sources": sources}
    

    return RAGChain(retriever, llm, PROMPT)


