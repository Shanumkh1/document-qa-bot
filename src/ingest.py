import os
import uuid
import chromadb
from pypdf import PdfReader
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from tqdm import tqdm

# ----------------------------
# CONFIG
# ----------------------------

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

DATA_DIR = "data"
DB_DIR = "db"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

COLLECTION_NAME = "document_knowledge_base"

# ----------------------------
# DOCUMENT LOADERS
# ----------------------------

def load_pdf(file_path):
    documents = []

    try:
        reader = PdfReader(file_path)

        for page_num, page in enumerate(reader.pages, start=1):

            text = page.extract_text()

            if text and text.strip():

                documents.append({
                    "text": " ".join(text.split()),
                    "metadata": {
                        "source": os.path.basename(file_path),
                        "page": page_num
                    }
                })

    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")

    return documents


def load_txt(file_path):

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    return [{
        "text": text,
        "metadata": {
            "source": os.path.basename(file_path),
            "page": 1
        }
    }]


def load_documents():

    all_documents = []

    for filename in os.listdir(DATA_DIR):

        file_path = os.path.join(DATA_DIR, filename)

        if filename.lower().endswith(".pdf"):

            print(f"Loading PDF: {filename}")

            all_documents.extend(load_pdf(file_path))

        elif filename.lower().endswith(".txt"):

            print(f"Loading TXT: {filename}")

            all_documents.extend(load_txt(file_path))

    return all_documents


# ----------------------------
# CHUNKING
# ----------------------------

def chunk_documents(
        documents,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP):

    chunks = []

    for doc in documents:

        text = doc["text"]
        metadata = doc["metadata"]

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end]

            chunks.append({
                "text": chunk_text,
                "metadata": metadata
            })

            start += (chunk_size - overlap)

    return chunks


# ----------------------------
# EMBEDDINGS
# ----------------------------

from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

def create_embeddings(texts):

    embeddings = embedding_model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True
    )

    return embeddings.tolist()


# ----------------------------
# VECTOR DATABASE
# ----------------------------

def save_to_chromadb(chunks):

    client = chromadb.PersistentClient(path=DB_DIR)

    try:
        client.delete_collection(COLLECTION_NAME)
    except:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME
    )

    texts = [chunk["text"] for chunk in chunks]

    metadatas = [chunk["metadata"] for chunk in chunks]

    ids = [str(uuid.uuid4()) for _ in chunks]

    embeddings = create_embeddings(texts)

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(f"\nIndexed {len(chunks)} chunks")
    print(f"Saved ChromaDB to '{DB_DIR}'")


# ----------------------------
# MAIN INDEXING PIPELINE
# ----------------------------

def run_indexing():

    print("\nLoading documents...")

    docs = load_documents()

    print(f"Loaded {len(docs)} pages")

    print("\nChunking documents...")

    chunks = chunk_documents(docs)

    print(f"Created {len(chunks)} chunks")

    print("\nSaving to ChromaDB...")

    save_to_chromadb(chunks)

    print("\nIndexing completed successfully!")


if __name__ == "__main__":
    run_indexing()