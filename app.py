import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama


# Page Settings
st.set_page_config(
    page_title="HR Handbook Assistant",
    page_icon="🤖",
    layout="wide"
)


# Title
st.title("🤖 HR Handbook Assistant")


# Chat History Memory
if "messages" not in st.session_state:
    st.session_state.messages = []


# Upload PDF
uploaded_file = st.file_uploader(
    "Upload HR PDF",
    type="pdf"
)


if uploaded_file:

    # Save uploaded PDF
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())


    # Load PDF
    loader = PyPDFLoader(uploaded_file.name)

    documents = loader.load()


    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = text_splitter.split_documents(documents)


    # Create Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


    # Create Vector Database
    vectorstore = Chroma.from_documents(
        chunks,
        embeddings
    )


    st.success("✅ HR PDF Processed Successfully!")


    # Show old messages
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])


    # Chat Input
    question = st.chat_input(
        "Ask HR related question..."
    )


    if question:

        # Save User Message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )


        # Display User Message
        with st.chat_message("user"):

            st.markdown(question)


        # Retrieve Similar Chunks
        docs = vectorstore.similarity_search(
            question,
            k=3
        )


        # Combine Context
        context = "\n".join(
            [doc.page_content for doc in docs]
        )


        # Local AI Model using Ollama
        llm = ChatOllama(
            model="phi3"
        )


        # Prompt
        prompt = f"""
You are an HR assistant.

Answer ONLY from the HR handbook context below.

If answer is not present in context, say:
"I could not find this information in the HR handbook."

Context:
{context}

Question:
{question}
"""


        # Generate Response
        response = llm.invoke(prompt)

        answer = response.content


        # Display Assistant Message
        with st.chat_message("assistant"):

            st.markdown(answer)


        # Save Assistant Message
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )


        # Show Source Chunks
        with st.expander("📄 Source Chunks"):

            for i, doc in enumerate(docs, start=1):

                st.markdown(f"### Chunk {i}")

                st.write(doc.page_content)

                st.write("------")