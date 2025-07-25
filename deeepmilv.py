import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.vectorstores import Milvus
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.messages import HumanMessage
from langchain.retrievers import BM25Retriever, EnsembleRetriever
import os
from tempfile import NamedTemporaryFile
import warnings

warnings.filterwarnings("ignore")

# Define the SYSTEM_TEMPLATE
SYSTEM_TEMPLATE = """
"You are a knowledgeable and helpful assistant designed to assist users effectively. "
"Your responses should draw from the provided content to craft a simple response. If the content "
"does not provide sufficient information, respond with 'There is no information corresponding to the given content. Therefore, it is difficult to answer.'"

<context>
{context}
</context>
"""

# Define the Question Answering Prompt Template
question_answering_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_TEMPLATE),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Initialize Azure OpenAI components
embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-3-small",
    azure_endpoint="https://amrxgenai.openai.azure.com/",
    api_key="14560021aaf84772835d76246b53397a",
    openai_api_version="2023-05-15"
)

llm = AzureChatOpenAI(
    api_key="14560021aaf84772835d76246b53397a",
    azure_endpoint="https://amrxgenai.openai.azure.com/",
    api_version="2024-02-15-preview",
    deployment_name="gpt"
)

# Milvus connection parameters
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "document_rag"

def process_file(file):
    """Process the uploaded file and return chunks"""
    # Save the file temporarily
    with NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_file_path = tmp_file.name
    
    # Load the file based on its extension
    if file.name.endswith('.pdf'):
        loader = PyPDFLoader(tmp_file_path)
    elif file.name.endswith('.docx'):
        loader = Docx2txtLoader(tmp_file_path)
    elif file.name.endswith('.txt'):
        loader = TextLoader(tmp_file_path)
    else:
        os.unlink(tmp_file_path)
        raise ValueError("Unsupported file format")
    
    documents = loader.load()
    
    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    
    # Clean up the temporary file
    os.unlink(tmp_file_path)
    
    return chunks

def store_in_milvus(chunks):
    """Store document chunks in Milvus vector database"""
    vector_db = Milvus.from_documents(
        chunks,
        embeddings,
        connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
        collection_name=COLLECTION_NAME
    )
    return vector_db, chunks

def create_ensemble_retriever(vector_db, chunks):
    """Create ensemble retriever combining vector and keyword search"""
    retriever_vectordb = vector_db.as_retriever(search_kwargs={"k": 5})
    keyword_retriever = BM25Retriever.from_documents(chunks)
    keyword_retriever.k = 5
    ensemble_retriever = EnsembleRetriever(
        retrievers=[retriever_vectordb, keyword_retriever],
        weights=[0.5, 0.5]
    )
    return ensemble_retriever

def main():
    st.title("Document RAG System with Milvus and Azure OpenAI")
    
    # Initialize session state variables
    if 'vector_db' not in st.session_state:
        st.session_state.vector_db = None
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'ensemble_retriever' not in st.session_state:
        st.session_state.ensemble_retriever = None
    if 'document_chain' not in st.session_state:
        st.session_state.document_chain = create_stuff_documents_chain(llm, question_answering_prompt)
    
    # File upload section
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a file", type=['pdf', 'docx', 'txt'])
    
    if uploaded_file is not None and not st.session_state.processed:
        with st.spinner("Processing document..."):
            try:
                chunks = process_file(uploaded_file)
                vector_db, chunks = store_in_milvus(chunks)
                st.session_state.vector_db = vector_db
                st.session_state.ensemble_retriever = create_ensemble_retriever(vector_db, chunks)
                st.session_state.processed = True
                st.success("Document processed and stored in Milvus successfully!")
                st.write(f"Number of chunks created: {len(chunks)}")
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    # Query section
    st.header("Query Your Document")
    user_input = st.text_input("Enter your question:")
    
    if user_input and st.session_state.processed:
        with st.spinner("Searching for answers..."):
            try:
                # Retrieve relevant documents
                docs = st.session_state.ensemble_retriever.invoke(user_input)
                
                # Display retrieved chunks
                st.subheader("Retrieved Document Chunks:")
                for i, doc in enumerate(docs, 1):
                    st.markdown(f"**Chunk {i}**")
                    st.write(doc.page_content)
                    st.write("---")
                
                # Generate answer using the document chain
                result = st.session_state.document_chain.invoke(
                    {
                        "context": docs,
                        "messages": [HumanMessage(content=user_input)],
                    }
                )
                
                # Display the final answer
                st.subheader("Answer:")
                st.write(result)
                
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")

if __name__ == "__main__":
    main()
