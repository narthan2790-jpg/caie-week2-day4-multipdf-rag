
import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title="Multi-PDF RAG Assistant")
st.title("Multi-PDF RAG Assistant")
st.write("Upload multiple PDFs and ask questions across all documents.")

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY environment variable is not set.")
    st.stop()

uploaded_files = st.file_uploader(
    "Upload multiple PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if st.button("Build Knowledge Base"):
    if not uploaded_files:
        st.warning("Please upload at least one PDF.")
    else:
        all_chunks = []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )

        with st.spinner("Reading and indexing PDFs..."):
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf"
                ) as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_path = temp_file.name

                try:
                    loader = PyPDFLoader(temp_path)
                    pages = loader.load()

                    for page in pages:
                        page.metadata["source"] = uploaded_file.name

                    chunks = splitter.split_documents(pages)
                    all_chunks.extend(chunks)
                finally:
                    os.remove(temp_path)

            if not all_chunks:
                st.error("No readable text found in the uploaded PDFs.")
            else:
                embeddings = OpenAIEmbeddings(api_key=api_key)
                st.session_state.store = FAISS.from_documents(
                    all_chunks, embeddings
                )
                st.session_state.indexed = True
                st.success(f"Indexed {len(all_chunks)} chunks.")

if st.session_state.get("indexed", False):
    st.subheader("Ask questions across your PDFs")

    question = st.text_input("Enter your question")

    if st.button("Get Answer"):
        if question.strip():
            with st.spinner("Searching documents..."):
                docs = st.session_state.store.as_retriever(
                    search_kwargs={"k": 5}
                ).invoke(question)

                context = "\n\n".join(
                    f"Source: {doc.metadata['source']}, "
                    f"Page: {doc.metadata['page'] + 1}\n{doc.page_content}"
                    for doc in docs
                )

                llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    api_key=api_key
                )

                prompt = f"""
                Answer the question using only the supplied PDF extracts.
                If the documents disagree, clearly explain the disagreement.
                If the answer is not available, say so.
                Include the relevant source filenames and page numbers.

                Question: {question}

                PDF extracts:
                {context}
                """

                response = llm.invoke(prompt)
                st.subheader("Answer")
                st.write(response.content)

                st.subheader("Retrieved Sources")
                for doc in docs:
                    st.caption(
                        f"{doc.metadata['source']} - "
                        f"Page {doc.metadata['page'] + 1}"
                    )
        else:
            st.warning("Please enter a question.")
