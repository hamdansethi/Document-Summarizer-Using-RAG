import os
import time
import getpass
from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path

# Set Google API key
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = "AIzaSyB4ihDBDj2_Y0hHTPOfoQHmth4Pq-DG208"

def load_document(file_path):
    """Load document based on file extension."""
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext in [".txt", ".md"]:
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file format. Use PDF, TXT, or Markdown.")
    return loader.load()

def chunk_document(documents):
    """Split documents into semantically meaningful chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_splitter.split_documents(documents)

def create_vector_store(documents):
    """Convert document chunks to embeddings and store in ChromaDB."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = Chroma(
        collection_name="summarization_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db"
    )
    document_ids = vector_store.add_documents(documents=documents)
    return vector_store

def setup_retrieval_chain(vector_store):
    """Set up retrieval and generation chain with Gemini."""
    llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
    
    prompt_template = """Use the following pieces of context to generate a concise and coherent summary of the document. Ensure the summary captures the main ideas and is fluent and accurate.

    Context:
    {context}

    Summary:"""
    
    prompt = PromptTemplate.from_template(prompt_template)
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever

def summarize_document(file_path):
    """Main function to process and summarize a document."""
    start_time = time.time()
    
    # Load and chunk document
    documents = load_document(file_path)
    chunks = chunk_document(documents)
    print(f"Split document into {len(chunks)} sub-documents.")
    
    # Create vector store
    vector_store = create_vector_store(chunks)
    
    # Set up retrieval and generation
    rag_chain, retriever = setup_retrieval_chain(vector_store)
    
    # Perform retrieval and generation
    query = "Summarize this document"
    summary = rag_chain.invoke(query)
    
    # Retrieve top-k contexts for display
    retrieved_docs = retriever.invoke(query)
    
    # Calculate metrics
    latency = time.time() - start_time
    similarity_scores = [doc.metadata.get("score", 0) for doc in retrieved_docs]
    
    # Prepare output
    output = {
        "summary": summary,
        "retrieved_contexts": [doc.page_content for doc in retrieved_docs],
        "latency": latency,
        "similarity_scores": similarity_scores
    }
    
    return output

def display_results(output):
    """Display the summary and retrieved contexts."""
    print("\n=== Document Summary ===")
    print(output["summary"])
    print("\n=== Retrieved Contexts ===")
    for i, context in enumerate(output["retrieved_contexts"], 1):
        print(f"Context {i}:\n{context}\n")
    print(f"Latency: {output['latency']:.2f} seconds")
    print("Similarity Scores:", output["similarity_scores"])

if __name__ == "__main__":
    # Example usage with sample documents
    sample_files = [
        "books.pdf"
    ]
    
    for file_path in sample_files:
        if os.path.exists(file_path):
            print(f"\nProcessing {file_path}...")
            output = summarize_document(file_path)
            display_results(output)
        else:
            print(f"\nFile {file_path} not found. Please provide valid document paths.")