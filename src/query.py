import os
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# ----------------------------
# CONFIG
# ----------------------------

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

DB_DIR = "db"
COLLECTION_NAME = "document_knowledge_base"

# ----------------------------
# EMBEDDING MODEL
# ----------------------------

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# ----------------------------
# RETRIEVAL
# ----------------------------

def retrieve_chunks(question, k=3):

    client = chromadb.PersistentClient(
        path=DB_DIR
    )

    collection = client.get_collection(
        COLLECTION_NAME
    )

    query_embedding = embedding_model.encode(
        question
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    return results


# ----------------------------
# ANSWER GENERATION
# ----------------------------

def generate_answer(question, retrieved_results):

    docs = retrieved_results["documents"][0]
    metas = retrieved_results["metadatas"][0]

    context = ""

    citations = []

    for doc, meta in zip(docs, metas):

        source = meta["source"]
        page = meta["page"]

        citations.append(
            f"{source} (Page {page})"
        )

        context += f"""
Source: {source}
Page: {page}

{doc}

--------------------------------
"""

    prompt = f"""
You are a document question answering assistant.

Use ONLY the supplied context.

If the answer is not found in the context, reply exactly:

I cannot find the answer in the provided documents.

Context:

{context}

Question:
{question}

Answer:
"""

    model = genai.GenerativeModel(
        "models/gemini-2.5-flash"
    )

    response = model.generate_content(
        prompt
    )

    answer = response.text

    return answer, citations


# ----------------------------
# MAIN QA FUNCTION
# ----------------------------

def ask_question(question):

    retrieved_results = retrieve_chunks(
        question
    )

    answer, citations = generate_answer(
        question,
        retrieved_results
    )

    return {
        "answer": answer,
        "citations": citations,
        "chunks": retrieved_results["documents"][0]
    }