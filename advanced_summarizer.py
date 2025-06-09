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

# Enable LangChain LLM cache
set_llm_cache(InMemoryCache())

# Load API keys from .env file
load_dotenv()
GOOGLE_API_KEYS = os.getenv("GOOGLE_API_KEYS", "").split(",")
api_key_index = 0

def set_api_key(index):
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEYS[index]

def set_next_api_key():
    global api_key_index
    api_key_index = (api_key_index + 1) % len(GOOGLE_API_KEYS)
    set_api_key(api_key_index)
    print(f"🔁 Switched to API Key #{api_key_index + 1}")

# Set initial API key
set_api_key(api_key_index)

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

def safe_invoke_rag(rag_chain, input_data, retries=5):
    for attempt in range(retries):
        try:
            return rag_chain.invoke(input_data)
        except Exception as e:
            print(f"⚠️ API Key #{api_key_index + 1} failed: {e}")
            if "quota" in str(e).lower() or "rate" in str(e).lower():
                set_next_api_key()
            else:
                raise e
    raise RuntimeError("❌ All API keys exhausted or failed.")

def summarize_document(file_path):
    start_time = time.time()

    docs = load_document(file_path)
    chunks = chunk_document_semantically(docs)
    print(f"📄 Document split into {len(chunks)} semantic chunks.")

    vector_store = create_vector_store(chunks)
    rag_chain, retriever = setup_rag_chain(vector_store)

    print("🧠 Running hierarchical summarization...")
    chunk_summaries = [
        safe_invoke_rag(rag_chain, chunk.page_content) for chunk in chunks
    ]
    summary = safe_invoke_rag(rag_chain, "\n\n".join(chunk_summaries))

    retrieved_docs = retriever.invoke("Summarize this document")
    similarity_scores = [doc.metadata.get("score", 0) for doc in retrieved_docs]
    latency = time.time() - start_time

    return {
        "summary": summary,
        "retrieved_contexts": [doc.page_content for doc in retrieved_docs],
        "latency": latency,
        "similarity_scores": similarity_scores
    }

def display_results(output):
    print("\n=== 📄 Document Summary ===\n")
    print(output["summary"])

    print("\n=== 📚 Retrieved Contexts ===\n")
    for i, context in enumerate(output["retrieved_contexts"], 1):
        print(f"--- Context {i} ---\n{context}\n")

    print(f"⚡ Latency: {output['latency']:.2f} seconds")
    print(f"🎯 Similarity Scores: {output['similarity_scores']}")

if __name__ == "__main__":
    sample_files = [
        "books.pdf"
    ]
    for file_path in sample_files:
        if os.path.exists(file_path):
            print(f"📂 Processing: {file_path}")
            output = summarize_document(file_path)
            display_results(output)
        else:
            print(f"❌ File not found: {file_path}")
