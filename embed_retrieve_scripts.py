import os
import re
from collections import defaultdict

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI

from rank_bm25 import BM25Okapi



def get_embeddings_file_name(tool="llamaparse", company_name="LOREAL_2023", chunking_method_name="chunk-pages",embedding_model_name="text-embedding-3-large"):
    return f"04-embeddings/{company_name}_chroma_db_ocr-{tool}_{chunking_method_name}_{embedding_model_name}"

    
    
# Function to create ChromaDB from document splits
def get_db_from_splits(splits, embeddings_model, save=False, file=None):
    if save:
        db = Chroma.from_documents(splits, embeddings_model, persist_directory=file)
    else:
        db = Chroma.from_documents(splits, embeddings_model)
    return db

# Function to load ChromaDB from a saved file
def get_db_from_file(file, embeddings_model):
    db = Chroma(persist_directory=file, embedding_function=embeddings_model)
    return db

# Function to get document embeddings
def get_embeddings(splits,company_name, tool, embeddings_model,chunking_method_name="",embedding_model_name="text-embedding-3-large"):
    embeddings_file_name = get_embeddings_file_name(tool, company_name,chunking_method_name,embedding_model_name)
    db = get_db_from_splits(splits, embeddings_model, save=True, file=embeddings_file_name)
    return db


def get_all_docs_from_db(db):
    # Fetch all document IDs
    all_docs = db.get(include=["documents", "metadatas"])
    
    # Extract documents and metadata
    documents = all_docs["documents"]
    metadatas = all_docs["metadatas"]

    
    if "metadatas" in all_docs:
        if (all_docs["metadatas"] == [None] * len(all_docs["metadatas"])):
        
            #doc_objects = [Document(page_content=doc) for doc in documents ]
            
            doc_objects = [Document(page_content=doc, metadata={'type':"text"}) for doc in documents ]

        
        else:
            doc_objects=[]
            for doc, meta in zip(documents, metadatas):
                if meta:
                    doc_objects.append(Document(page_content=doc, metadata=meta))
                else:
                    doc_objects.append(Document(page_content=doc, metadata={'type':"text"}))
    else:
        doc_objects = [Document(page_content=doc, metadata={'type':"text"}) for doc in documents ]
    
    return doc_objects


# 🔹 NEW: Create BM25 index
def create_bm25_index(splits):
    """
    Create a BM25 index from document splits.
    """
    tokenized_corpus = [doc.page_content.lower().split() for doc in splits]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, splits

# 🔹 NEW: BM25 Retrieval Function
def bm25_retrieve(query, bm25, docs, top_k=5):
    """
    Retrieve top-k documents using BM25.
    """
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    
    # Sort by BM25 score
    top_doc_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    
    return [docs[i] for i in top_doc_indices]



# 🔹 NEW: Combine BM25 + Vector Search
def hybrid_retriever(query, retriever, bm25, bm25_docs, k=5):
    """
    Retrieve `k/2` documents from BM25 and `k/2` from vector search.
    """
    k_bm25 = max(1, k // 2)
    k_vector = k - k_bm25  # Ensure the total remains k
    
    # Get BM25 results
    bm25_results = bm25_retrieve(query, bm25, bm25_docs, top_k=k_bm25)
    
    # Get vector search results
    vector_results = retriever.get_relevant_documents(query)
    
    # Merge results while removing duplicates
    unique_results  ={doc.page_content: doc for doc in (vector_results+bm25_results)}
    
    #unique_results={doc.page_content: doc for doc in bm25_results}
    
    return list(unique_results.values())



def get_vector_retriever(query, retriever,k=8):
    vector_results = retriever.get_relevant_documents(query)
    
    # Merge results while removing duplicates
    unique_results = {doc.page_content: doc for doc in vector_results}
    
    return list(unique_results.values())


# Function to prepare additional information
def prepare_additional_info(**kwargs):
    additional_info = ""
    for key, value in kwargs.items():
        additional_info += f"{key}: {value}\n"
    return additional_info.strip()




def format_docs(docs):
    """
    Check if any document has 'table' metadata and add it to the context.
    """
    context = ""
    i=0
    for doc in docs:
        i=i+1
        context += f"Context Num-{str(i)} ==> \n{doc.page_content}\n\n"
        # Check if the document contains metadata named "table"
        #if "table" in doc.metadata:
            # Add the table metadata to the context
            #context += f"For more information see the Table: {doc.metadata['table']}\n"
        # Append the regular page content
        
    
    i=0
    tab=False
    for doc in docs:
        
        i=i+1
        
        # Check if the document contains metadata named "table"
        if "table" in doc.metadata:
            if not tab:
                context += "\n\n ANNEXE: \n\n"
                tab=True
            # Add the table metadata to the context
            context += f"For more information on Context Num-{str(i)} , see the Table:\n {doc.metadata['table']}\n\n"
        # Append the regular page content

        
    return context