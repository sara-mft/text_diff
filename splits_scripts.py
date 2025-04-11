import os
import re
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import Document
from openai import AzureOpenAI
# 1. Import the necessary library
from openai import OpenAI


import time




import json
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain.schema import Document


def is_only_headers(text):
    """
    Check if the given text contains only Markdown headers with no additional content.
    """
    lines = text.strip().split("\n")
    
    # Regex to match Markdown headers (#, ##, ###, etc.)
    header_pattern = re.compile(r"^#{1,6}\s+\S+")
    
    # Return True if all lines match the header pattern, False otherwise
    return all(header_pattern.match(line) for line in lines if line.strip())


def get_markdown_splits(json_file_path, chunk_size=1000, chunk_overlap=200):
    """
    Splits text from a JSON file into chunks while retaining page number metadata.

    Args:
        json_file_path (str): Path to the JSON file.
        chunk_size (int): Character limit per chunk.
        chunk_overlap (int): Number of overlapping characters between chunks.

    Returns:
        List[Document]: List of Document chunks with page metadata.
    """
    with open(json_file_path, "r", encoding="utf-8") as f:
        page_data = json.load(f)

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3")
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )
    
    if chunk_size!=None and chunk_overlap!=None:

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    all_chunks = []

    for page in page_data:
        page_number = page.get("page_number")
        content = page.get("content", "")
        

        # Remove all "### END TABLE" markers
        content = content.replace("### END TABLE", "")

        
        
        if not content.strip():
            continue

        # Split based on markdown headers first
        header_docs = markdown_splitter.split_text(content)
        
        if chunk_size==None or chunk_overlap==None:
            for h in header_docs: 
                h.metadata["page_number"] = page_number
                h.metadata["page_content"] =content
                all_chunks.append(h) 
            
        else:
            
            for i, doc in enumerate(header_docs):
                if "Header 3" in doc.metadata and doc.metadata["Header 3"] == "BEGIN TABLE":
                    doc.metadata["page_number"] = page_number
                    doc.metadata["page_content"] =content
                    all_chunks.append(doc)
            
                else:
                    if not is_only_headers(doc.page_content): 
                        # Then split into smaller chunks
                        chunks = text_splitter.split_documents(doc)

                        # Attach page number as metadata
                        for chunk in chunks:
                            chunk.metadata["page_number"] = page_number
                            chunk.metadata["page_content"] =content
                            all_chunks.append(chunk)

    return all_chunks



def get_page_level_chunks(json_file_path):
    """
    Loads a JSON file where each item is a page and returns one chunk per page with page number metadata.
    Removes '### END TABLE' from each page content.

    Args:
        json_file_path (str): Path to the JSON file in format:
                              [{"page_number": 1, "content": "..."}, ...]

    Returns:
        List[Document]: List of Documents with page-level content.
    """
    with open(json_file_path, "r", encoding="utf-8") as f:
        page_data = json.load(f)

    page_chunks = []

    for page in page_data:
        page_number = page.get("page_number")
        content = page.get("content", "").replace("### END TABLE", "").strip()
        content = page.get("content", "").replace("### BEGIN TABLE", "").strip()

        if not content:
            continue

        doc = Document(
            page_content=content,
            metadata={"page_number": page_number}
        )
        page_chunks.append(doc)

    return page_chunks




def get_recursive_chunks_per_page(
    json_file_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
):
    """
    Recursively chunks each page's content separately and keeps page_number metadata.

    Args:
        json_file_path (str): Path to JSON with page-level content.
        chunk_size (int): Max characters per chunk.
        chunk_overlap (int): Overlap between chunks.

    Returns:
        List[Document]: List of Document chunks with page_number metadata.
    """
    with open(json_file_path, "r", encoding="utf-8") as f:
        page_data = json.load(f)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    all_chunks = []

    for page in page_data:
        page_number = page.get("page_number")
        content = page.get("content", "").replace("### END TABLE", "").strip()

        if not content:
            continue

        chunks = splitter.split_text(content)
        for chunk in chunks:
            all_chunks.append(Document(
                page_content=chunk.strip(),
                metadata={"page_number": page_number, "page":content}
            ))

    return all_chunks









































# Fonction pour obtenir les découpes de texte à partir d'un fichier PDF converti en texte
def get_splits(text_file):
        with open(text_file, "r", encoding="utf-8") as f:
            documents = f.read()
        headers_to_split_on = [("#", "Header 1"),("##", "Header 2"),("###", "Header 3")]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
        md_header_splits = markdown_splitter.split_text(documents)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        splits = text_splitter.split_documents(md_header_splits)
        return splits



# 2. Define the function (mostly unchanged from your original version)
def summarize_table_2(client: OpenAI, previous_content: str, table_content: str, next_content: str) -> str:
    """
    Summarizes a Markdown table using OpenAI GPT-4o with context from the
    previous and next chunks. Uses the standard OpenAI API client.
    """

    # Construct the prompt (same logic as before)
    prompt = f"""

Si le contenu est un tableau, résumez-le de manière concise dans la même langue que celle du tableau.

Vous devez citer tous les libellés, noms des colonnes et des lignes, ainsi que les années.

Contexte :

Contenu précédent : {previous_content[:500]}... (contexte fourni avant le tableau)

Contenu du tableau : {table_content}

Contenu suivant : {next_content[:500]}... (contexte fourni après le tableau)

Assurez-vous que seul le tableau est résumé. Les parties précédentes et suivantes ne sont fournies qu'à titre de contexte et ne doivent pas être résumées.

Fournissez le résumé en un maximum de 10 phrases.

Si le contenu est une image ou un texte brut, retournez exactement le même contenu en entrée.

Identifiez la langue du texte dans l'image et mentionnez-la au début.

Le résumé doit être dans la même langue que celle du texte dans l'image.

"""

    # Call OpenAI API to summarize the table
    # Ensure the client passed is an instance of openai.OpenAI
    # The model ID "gpt-4o" is used directly.
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an expert in analyzing tabular data."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1500,
            temperature=0.0,
            top_p=1.0,
            model="gpt-4o",  # Use the standard OpenAI model ID
        )
        # Return the summary generated by GPT-4o
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"An error occurred while calling the OpenAI API: {e}")
        # Handle the error appropriately - maybe return the original table or an error message
        # For the requirement "If the content is an image or plain text, reurn exactly the same input content"
        # we might assume an error means we can't process, so return original? Or signal error?
        # Let's return an error message for now, or you could return table_content
        return f"Error summarizing table: {e}"

def summarize_table(client, previous_content, table_content, next_content):
    """Summarizes a Markdown table using Azure OpenAI GPT-4o with context from the previous and next chunks."""
    
    # Construct the prompt to ensure only the table is summarized, while previous and next chunks provide context
    prompt = f"""If the content is a Markdown table, summarize it concisely in the same language as the language of the table.
    - You have to cite all labels, columns' and rows' names, and years.

    Context:
    - Previous content: {previous_content[:500]}... (context provided before the table)
    - Table content: {table_content}
    - Next content: {next_content[:500]}... (context provided after the table)

    Please ensure that only the table is summarized. The previous and next chunks are only for context, and should not be summarized themselves.
    
    Provide the summary in a maximum of 10 sentences.
    
    - If the content is an image or plain text, reurn exactly the same input content
    """

    # Call Azure OpenAI API to summarize the table
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are an expert in analyzing tabular data."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=3000,
        temperature=0.0,
        top_p=1.0,
        model="gpt-4o",
    )

    # Return the summary generated by GPT-4o
    return response.choices[0].message.content.strip()




def is_only_headers(text):
    """
    Check if the given text contains only Markdown headers with no additional content.
    """
    lines = text.strip().split("\n")
    
    # Regex to match Markdown headers (#, ##, ###, etc.)
    header_pattern = re.compile(r"^#{1,6}\s+\S+")
    
    # Return True if all lines match the header pattern, False otherwise
    return all(header_pattern.match(line) for line in lines if line.strip())

def get_splits_2(client,text_file, chunk_size=1000, chunk_overlap=100):
    with open(text_file, "r", encoding="utf-8") as f:
        documents = f.read()

    # Define Markdown headers for splitting
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    
    # Split the document based on markdown headers
    md_header_splits = markdown_splitter.split_text(documents)

    # RecursiveCharacterTextSplitter for fine-grained chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    final_splits = []
    previous_chunk = None  # To hold the previous chunk for later use

    for i, doc in enumerate(md_header_splits):
        # Skip chunks that contain only headers


        # If "Header 3" is "BEGIN TABLE", summarize the table with preceding and next chunks
        if "Header 3" in doc.metadata and doc.metadata["Header 3"] == "BEGIN TABLE":
            # Get the previous and next chunks
            #previous_content = previous_chunk.page_content if previous_chunk else ""
            #next_content = md_header_splits[i + 1].page_content if i + 1 < len(md_header_splits) else ""
            
            # Call the summarize_table function with previous, table, and next content
            #table_summary = summarize_table(previous_content, doc.page_content, next_content)

            # Store the table summary in the content and the original table in the metadata
            #doc.metadata["table"] = doc.page_content  # Storing the original table content in the metadata
            #doc.page_content = table_summary
            
            final_splits.append(doc)
            #time.sleep(2)
        else:
            # Otherwise, apply RecursiveCharacterTextSplitter to other chunks
            
            
            
            if not is_only_headers(doc.page_content):    
                split_chunks = text_splitter.split_documents([doc])
                final_splits.extend(split_chunks)

        # Update the previous chunk for the next iteration
        previous_chunk = doc

    return final_splits



def get_splits_3(client,text_file, chunk_size=1000, chunk_overlap=100):
    with open(text_file, "r", encoding="utf-8") as f:
        documents = f.read()

    # Define Markdown headers for splitting
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    
    # Split the document based on markdown headers
    md_header_splits = markdown_splitter.split_text(documents)

    # RecursiveCharacterTextSplitter for fine-grained chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    final_splits = []
    previous_chunk = None  # To hold the previous chunk for later use

    for i, doc in enumerate(md_header_splits):
        # Skip chunks that contain only headers


        # If "Header 3" is "BEGIN TABLE", summarize the table with preceding and next chunks
        if "Header 3" in doc.metadata and doc.metadata["Header 3"] == "BEGIN TABLE":
            # Get the previous and next chunks
            previous_content = previous_chunk.page_content if previous_chunk else ""
            next_content = md_header_splits[i + 1].page_content if i + 1 < len(md_header_splits) else ""
            
            # Call the summarize_table function with previous, table, and next content
            table_summary = summarize_table_2(client,previous_content, doc.page_content, next_content)

            # Store the table summary in the content and the original table in the metadata
            doc.metadata["table"] = doc.page_content  # Storing the original table content in the metadata
            doc.page_content = table_summary
            
            final_splits.append(doc)
            #time.sleep(5)
        else:
            # Otherwise, apply RecursiveCharacterTextSplitter to other chunks
            
            
            
            if not is_only_headers(doc.page_content):    
                split_chunks = text_splitter.split_documents([doc])
                final_splits.extend(split_chunks)

        # Update the previous chunk for the next iteration
        previous_chunk = doc

    return final_splits