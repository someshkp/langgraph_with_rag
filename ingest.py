
import glob
import os

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

load_dotenv()

DOCS_DIR = os.path.join(os.path.dirname(__file__), "sample_docs")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "faiss_index")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def load_and_split():
    """Load .txt files and split into ~500-char chunks with overlap."""
    chunks = []
    for path in glob.glob(os.path.join(DOCS_DIR, "*.txt")):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        
        source = os.path.basename(path)
        paragraph= [p.strip() for p in text.split('\n\n') if p.strip()]
        
        current=""
        for p in paragraph:
            if len(current) + len(p) < CHUNK_SIZE:
                current += "\n\n" + p
            else:
                if current:
                    chunks.append(Document(page_content=current.strip(), metadata={"source": source}))
                current = p
        if current:
            chunks.append(Document(page_content=current.strip(), metadata={"source": source}))
    return chunks


def main():
    print("Loading and splitting documents...")
    chunks = load_and_split()
    print(f"Split into {len(chunks)} chunks.")
    print("Embedding and building FAISS index...")
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    print("Creating vector store...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("Saving vector store...")
    vectorstore.save_local(INDEX_DIR)
    print("Done.")


if __name__ == "__main__":
    main()