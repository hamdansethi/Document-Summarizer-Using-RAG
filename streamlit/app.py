import streamlit as st
import tempfile
import os
from summarizer import summarize

st.set_page_config(page_title="📄 Advanced Document Summarizer", layout="wide")
st.title("📚 Advanced Document Summarizer")
st.markdown("Upload a `.pdf`, `.txt`, or `.md` file. The app will extract, chunk, and summarize it intelligently using Gemini + LangChain.")

uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt", "md"])

if uploaded_file is not None:
    file_suffix = os.path.splitext(uploaded_file.name)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("🔍 Processing..."):
        result = summarize(tmp_path)

    if result.get("error"):
        st.error(f"❌ Error: {result['error']}")
    else:
        st.success("✅ Summary generated!")

        st.subheader("📄 Summary")
        st.write(result["summary"])

        st.subheader("📚 Retrieved Contexts")
        for i, context in enumerate(result["contexts"], 1):
            with st.expander(f"Context {i}"):
                st.write(context)

        st.sidebar.markdown("### ℹ️ Stats")
        st.sidebar.write(f"**Chunks:** {result['chunk_count']}")
        st.sidebar.write(f"**Latency:** {result['latency']:.2f} seconds")
        st.sidebar.write(f"**Similarity Scores:** {result['similarity_scores']}")
