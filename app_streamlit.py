import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

# ---------- CONFIGURATION ----------
st.set_page_config(page_title="CSRD RAG Assistant", layout="wide", page_icon="📘")

# ---------- SIDEBAR ----------
st.sidebar.title("🔧 Configuration")
company_name = st.sidebar.text_input("Company", "LOREAL_2023")

ocr_tool = st.sidebar.selectbox("OCR Tool", ["pypdf", "pymupdf", "azure", "llm2_gpt-4o"])
chunking_method_name = st.sidebar.selectbox("Chunking Method", ["chunk-pages", "chunk-markdown", "chunk-recursive"])
embedding_model_name = st.sidebar.text_input("Embedding Model", "embeddings-multilingual-e5-large-instruct")
top_k = st.sidebar.slider("Top K Results", 3, 12, 8)
model_name = "gpt-4o"


st.sidebar.markdown("### 🧠 Prompt Settings")

global_prompt = st.sidebar.text_area(
    "System Prompt (global behavior)",
    value="You are an expert in gathering information from CSRD annual reports.",
    height=100
)

global_instructions = st.sidebar.text_area(
    "Global Instructions (prepended to context)",
    value=(
        "- S'assurer de bien comprendre la nature de la question pour fournir une réponse pertinente.\n"
        "- Si la question ne précise pas l'année, la région, donnez les informations sur plusieurs années ou régions.\n"
        "- Si la réponse ne se trouve pas dans les documents fournis, ne l'inventez pas, dites que vous ne savez pas."
    ),
    height=120
)







# ---------- AZURE CONFIG ----------
AZURE_ENDPOINT = ""
AZURE_KEY = ""

client_azure = ChatCompletionsClient(
    endpoint=AZURE_ENDPOINT,
    credential=AzureKeyCredential(AZURE_KEY),
)

# ---------- HELPER FUNCTIONS ----------
def get_embeddings_file_name():
    return f"04-embeddings/{company_name}_chroma_db_ocr-{ocr_tool}_{chunking_method_name}_{embedding_model_name}"

def get_db_from_file(file, embeddings_model):
    db = Chroma(persist_directory=file, embedding_function=embeddings_model)
    return db

def get_all_docs_from_db(db):
    all_docs = db.get(include=["documents", "metadatas"])
    documents = all_docs["documents"]
    metadatas = all_docs["metadatas"]
    doc_objects = []

    for doc, meta in zip(documents, metadatas):
        meta = meta if meta else {'type': 'text'}
        doc_objects.append(Document(page_content=doc, metadata=meta))
    
    return doc_objects

def get_vector_retriever(query, retriever):
    vector_results = retriever.get_relevant_documents(query)
    unique_results = {doc.page_content: doc for doc in vector_results}
    return list(unique_results.values())

def query_azure(question, context, global_prompt, global_instructions, question_specific_instructions):
    context_text = "\n".join([doc.page_content for doc in context])
    
    messages = [
        SystemMessage(content=global_prompt),
        UserMessage(content=f"{global_instructions}"),
        UserMessage(content=f"\nAnswer the following question:\n{question}\n Instructions: \n{question_specific_instructions}"),
        UserMessage(content=f"\nContext:\n{context_text}")
    ]
    
    response = client_azure.complete(
        messages=messages,
        temperature=0.0,
        top_p=1.0,
        model=model_name
    )
    return response.choices[0].message.content

# ---------- EMBEDDINGS ----------
embeddings_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct")

# ---------- MAIN APP ----------
st.title("📘 CSRD RAG Assistant")
st.markdown("Ask a question based on the company's annual report and get a contextualized answer with sources.")




question = st.text_area("Your Question", height=100, placeholder="E.g., Quelles sont les entités ou personnes qui composent l'actionnariat de LOREAL ?")


question_specific_instructions = st.text_area(
    "Question-Specific Instructions (optional)",
    placeholder="E.g., If possible, format the answer as a markdown table.",
    height=80
)


if st.button("Get Answer"):
    with st.spinner("Fetching information and generating answer..."):
        emb_file_name = get_embeddings_file_name()
        db = get_db_from_file(emb_file_name, embeddings_model=embeddings_model)
        vector_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
        context_chunks = get_vector_retriever(question, vector_retriever)
        answer = query_azure(question, context_chunks ,global_prompt, global_instructions, question_specific_instructions)

    st.success("✅ Answer Generated")

    st.markdown("### ✨ Answer")
    st.markdown(f"<div style='padding: 1em; background-color: #f0f2f6; border-left: 6px solid #2c7be5;'>{answer}</div>", unsafe_allow_html=True)

    st.markdown("### 📚 Sources")
    for i, chunk in enumerate(context_chunks):
        with st.expander(f"📄 Source {i+1} — Page {chunk.metadata.get('page_number', 'N/A')}"):
            st.markdown(f"<div style='background-color:#f9f9f9;padding:10px;border-left:5px solid #91caff;'>{chunk.page_content}</div>", unsafe_allow_html=True)
