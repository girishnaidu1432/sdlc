import os
import pickle
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import Milvus
from pymilvus import connections
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
import tempfile
import warnings

warnings.filterwarnings("ignore")

# ---------- Azure OpenAI Configuration ----------
AZURE_API_KEY = "14560021aaf84772835d76246b53397a"
AZURE_ENDPOINT = "https://amrxgenai.openai.azure.com/"
AZURE_DEPLOYMENT_NAME = "gpt"
AZURE_API_VERSION = "2024-02-15-preview"
EMBEDDING_MODEL = "text-embedding-3-small"

# ---------- Milvus Configuration ----------
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"

# ---------- Streamlit UI ----------
st.set_page_config(page_title="RAG with Milvus", layout="wide")
st.title("🔍 RAG PDF Chat using Milvus + Azure OpenAI")

uploaded_file = st.file_uploader("📄 Upload a PDF", type="pdf")

if uploaded_file:
    with st.spinner("Processing PDF..."):
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_pdf_path = tmp_file.name

        # Extract filename (without extension) for use as collection name
        base_filename = os.path.splitext(uploaded_file.name)[0]
        pkl_file = f"{base_filename}.pkl"

        # Step 1: Load and split PDF
        loader = PyPDFLoader(temp_pdf_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=500)
        splits = text_splitter.split_documents(docs)

        # Save to pickle
        if os.path.exists(pkl_file):
            with open(pkl_file, "rb") as file:
                current_splits = pickle.load(file)
                st.info("Existing pickle file found. Appending new splits.")
        else:
            current_splits = []
            st.info("New pickle file created.")

        current_splits.extend(splits)
        with open(pkl_file, "wb") as file:
            pickle.dump(current_splits, file)

        # Step 2: Embedding
        embeddings = AzureOpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
            openai_api_version="2023-05-15"
        )

        # Step 3: Milvus Connection + Embedding Upload
        connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
        vectorstore = Milvus.from_documents(
            documents=splits,
            embedding=embeddings,
            collection_name=base_filename,
            connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT}
        )
        st.success("✅ PDF embedded and stored in Milvus!")

        # Step 4: Setup LLM
        llm = AzureChatOpenAI(
            api_key=AZURE_API_KEY,
            azure_endpoint=AZURE_ENDPOINT,
            api_version=AZURE_API_VERSION,
            deployment_name=AZURE_DEPLOYMENT_NAME
        )

        # Step 5: Prompt + Document QA Chain
        SYSTEM_TEMPLATE = """
        You are a helpful assistant. Use the provided <context> to answer user questions.
        If insufficient info is found, respond with:
        'There is no information corresponding to the given content. Therefore, it is difficult to answer.'
        <context>
        {context}
        </context>
        """
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_TEMPLATE),
            MessagesPlaceholder(variable_name="messages"),
        ])
        document_chain = create_stuff_documents_chain(llm, qa_prompt)

        # Step 6: Setup retrievers
        keyword_retriever = BM25Retriever.from_documents(current_splits)
        keyword_retriever.k = 5
        vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, keyword_retriever],
            weights=[0.5, 0.5]
        )

        # Step 7: Chat Interface
        st.subheader("💬 Ask a question from your uploaded PDF")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        query = st.text_input("Ask a question:")

        if st.button("Submit Query") and query:
            with st.spinner("Thinking..."):
                docs = ensemble_retriever.invoke(query)

                result = document_chain.invoke({
                    "context": docs,
                    "messages": [HumanMessage(content=query)],
                })

                # Save chat history
                st.session_state.chat_history.append(("user", query))
                st.session_state.chat_history.append(("assistant", result['content']))

        # Display Chat History
        if st.session_state.chat_history:
            st.divider()
            st.subheader("📜 Chat History")
            for speaker, msg in st.session_state.chat_history:
                if speaker == "user":
                    st.markdown(f"**👤 You:** {msg}")
                else:
                    st.markdown(f"**🤖 Assistant:** {msg}")
