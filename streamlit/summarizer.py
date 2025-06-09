import os
import time
from pathlib import Path
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache

# Load API keys from .env
load_dotenv()
API_KEYS = os.getenv("GOOGLE_API_KEYS", "").split(",")
API_KEY_INDEX = 0

set_llm_cache(InMemoryCache())

def rotate_key():
    global API_KEY_INDEX
    API_KEY_INDEX = (API_KEY_INDEX + 1) % len(API_KEYS)
    os.environ["GOOGLE_API_KEY"] = API_KEYS[API_KEY_INDEX]

rotate_key()

def load_document(file_path):
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext in [".txt", ".md"]:
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file format.")
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = Path(file_path).name
    return docs

def chunk_document_semantically(documents):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    splitter = SemanticChunker(embeddings=embeddings)
    return splitter.split_documents(documents)

def create_vector_store(documents):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = Chroma(
        collection_name="advanced_rag_summary",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db"
    )
    vector_store.add_documents(documents)
    return vector_store

def setup_rag_chain(vector_store):
    llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

    prompt_template = """
You are an expert summarizer. Given the following relevant content chunks, write a fluent, structured, and concise summary. Avoid repetition and focus on key ideas.

Context:
{context}

Summary:
"""
    prompt = PromptTemplate.from_template(prompt_template)

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "lambda_mult": 0.5}
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever

def summarize(file_path):
    start_time = time.time()

    try:
        docs = load_document(file_path)
        chunks = chunk_document_semantically(docs)
        vector_store = create_vector_store(chunks)
        rag_chain, retriever = setup_rag_chain(vector_store)

        chunk_summaries = [rag_chain.invoke(chunk.page_content) for chunk in chunks]
        final_summary = rag_chain.invoke("\n\n".join(chunk_summaries))

        retrieved_docs = retriever.invoke("summarize this document")
        similarity_scores = [doc.metadata.get("score", 0) for doc in retrieved_docs]
        latency = time.time() - start_time

        return {
            "summary": final_summary,
            "contexts": [doc.page_content for doc in retrieved_docs],
            "similarity_scores": similarity_scores,
            "latency": latency,
            "chunk_count": len(chunks)
        }

    except Exception as e:
        rotate_key()
        return {"error": str(e)}
