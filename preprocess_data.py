
import os
import pandas as pd
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings


# Load the full dataset
csv_path = "mtsamples.csv"
print(f"Loading dataset from {csv_path}...")
df = pd.read_csv(csv_path)
df = df.dropna(subset=["transcription"])
print(f"Total records loaded: {len(df)}")


# Create LangChain documents from each record
print("Creating documents...")
documents = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    content = (
        f"[Specialty: {row['medical_specialty']}] "
        f"[Description: {row['description']}] "
        f"{row['transcription']}"
    )
    doc = Document(
        page_content=content,
        metadata={
            "source": row.get("sample_name", "unknown"),
            "specialty": row["medical_specialty"],
            "description": row["description"]
        }
    )
    documents.append(doc)
print(f"Documents created: {len(documents)}")


# Split documents into chunks for retrieval
print("Splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)
print(f"Chunks created: {len(chunks)}")


# Create FAISS index from chunks
print("Building FAISS index...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)

# Save the vectorstore
output_dir = "vectorstore"
os.makedirs(output_dir, exist_ok=True)
vectorstore.save_local(output_dir)
print(f"Vectorstore saved to '{output_dir}/'")
print("Preprocessing complete.")
