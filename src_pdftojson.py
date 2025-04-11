from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence.models import AnalyzeResult, DocumentContentFormat
from azure.ai.documentintelligence.models import AnalyzeResult, DocumentPage

from PyPDF2 import PdfReader

from openai import OpenAI
import base64
import unstructured_client
from unstructured.partition.pdf import partition_pdf
from openai import OpenAIError


from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage, TextContentItem, ImageContentItem, ImageUrl
from azure.core.credentials import AzureKeyCredential

import io
import pandas as pd
import re

import json

import fitz  # PyMuPDF


import os
import base64
import io
import argparse # For command-line arguments
from PIL import Image
from pdf2image import convert_from_path, pdfinfo_from_path
from openai import OpenAI, APIError
from dotenv import load_dotenv
from typing import List, Dict, Optional

from llama_parse import LlamaParse




def extract_pages(input_pdf, output_pdf, start_page, end_page):
    doc = fitz.open(input_pdf)
    new_doc = fitz.open()

    # Pages are 0-indexed in PyMuPDF, so subtract 1 from start_page
    new_doc.insert_pdf(doc, from_page=start_page - 1, to_page=end_page - 1)

    new_doc.save(output_pdf)
    new_doc.close()
    doc.close()



################################ PYMuPDF ####################################################################



def extract_text_with_pymupdf_to_json(pdf_path: str, json_output_path: str) -> None:
    """
    Extract text from a PDF using PyMuPDF and save it in a structured JSON format.

    Output JSON format:
    {
      "pages": [
        {"page_number": 1, "content": "..."},
        {"page_number": 2, "content": "..."}
      ]
    }

    Args:
        pdf_path (str): Full path to the input PDF file.
        json_output_path (str): Destination path for the resulting JSON file.
    """
    content = {"pages": []}

    try:
        doc = fitz.open(pdf_path)  # Open the PDF using PyMuPDF
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()  # Extract plain text
            content["pages"].append({
                "page_number": page_num,
                "content": text.strip()
            })
        doc.close()
    except Exception as e:
        print(f"❌ Error reading PDF '{pdf_path}': {e}")
        return

    try:
        os.makedirs(os.path.dirname(json_output_path), exist_ok=True)
        with open(json_output_path, "w", encoding="utf-8") as json_file:
            json.dump(content, json_file, indent=2, ensure_ascii=False)
        print(f"✅ Text extracted and saved to JSON: {json_output_path}")
    except Exception as e:
        print(f"❌ Error writing JSON file: {e}")



################################ PYPDF ####################################################################

def pdf_to_json_pypdf(path_to_sample_documents, output_json_file_name):
    pdf_reader = PdfReader(path_to_sample_documents)
    content = {
        "pages": []
    }

    for idx, page in enumerate(pdf_reader.pages):
        page_text = page.extract_text()
        content["pages"].append({
            "page_number": idx + 1,
            "content": page_text.strip() if page_text else ""
        })

    with open(output_json_file_name, "w", encoding="utf-8") as f:
        json.dump(content, f, indent=2, ensure_ascii=False)








################################# LLAMAPARSE #######################################################""



def extract_text_from_pdf_llamaparse_json(
    pdf_path: str,
    json_output_path: str,
    result_type: str = "markdown",
    parse_mode: str = "parse_page_with_lvm",
    vendor_model: str = "openai-gpt-4o-mini",
    verbose: bool = False
) -> None:
    """
    Uses LlamaParse with multimodal LLM to extract structured Markdown from a PDF and saves it as JSON.

    Args:
        pdf_path (str): Path to the PDF file.
        json_output_path (str): Path to save the JSON output.
        result_type (str): Output format, default is 'markdown'.
        parse_mode (str): Page-level parsing mode.
        vendor_model (str): Multimodal model used (e.g., GPT-4o).
        verbose (bool): Print progress logs if True.
    """
    parser = LlamaParse(
        result_type=result_type,
        parse_mode=parse_mode,
        vendor_multimodal_model_name=vendor_model,
        # vendor_multimodal_api_key=os.getenv("YOUR_KEY")  # Optional if needed
    )

    try:
        documents = parser.load_data(pdf_path)
        if verbose:
            print(f"Parsed {len(documents)} pages from {pdf_path}")
    except Exception as e:
        print(f"Failed to parse PDF: {e}")
        return

    page_data = []
    for idx, doc in enumerate(documents):
        page_number = idx + 1
        content = doc.text.strip()
        if verbose:
            print(f"Extracted page {page_number}: {len(content)} characters")
        page_data.append({
            "page_number": page_number,
            "content": content
        })

    try:
        os.makedirs(os.path.dirname(json_output_path), exist_ok=True)
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(page_data, f, ensure_ascii=False, indent=2)
        print(f"✅ Output saved to {json_output_path}")
    except Exception as e:
        print(f"Error writing JSON file: {e}")




################################# AZURE DOCUMENT INTELLIGENCE #######################################################""


def extract_number(s: str) -> int:
    """Extracts a numeric ID from a paragraph reference string.
    
    Args:
        s (str): Input string containing the paragraph reference.
    
    Returns:
        int: Extracted numeric ID.
    
    Raises:
        ValueError: If no valid number is found in the string.
    """
    
    PARAGRAPH_REGEX = re.compile(r'/paragraphs/(\d+)')
    match = PARAGRAPH_REGEX.search(s)
    if match:
        return int(match.group(1))
    
    raise ValueError(f"Invalid format: No paragraph number found in '{s}'")
    
    
def extract_tables_markdown(result: AnalyzeResult, page: DocumentPage):
    """Extracts tables from a document and converts them into Markdown format.
    
    Args:
        result (AnalyzeResult): The result of document analysis.
        page (DocumentPage): The page to extract tables from.
    
    Returns:
        tuple: A tuple containing:
            - List of tables in Markdown format.
            - List of first paragraph positions.
            - List of last paragraph positions.
    """
    if not result.tables:
        return [], [], []

    tables, first_positions, last_positions = [], [], []

    for table in result.tables:
        if table.bounding_regions[0].page_number != page.page_number:
            continue  # Skip tables that are not on the current page
        
        # Extract paragraph positions for the table
        list_table_paragraphs = [
            extract_number(el.elements[0]) for el in table.cells if el.elements
        ]

        if not list_table_paragraphs:
            continue  # Skip tables without valid paragraph references

        first_positions.append(list_table_paragraphs[0])
        last_positions.append(list_table_paragraphs[-1])

        # Create DataFrame from table content
        data = [[""] * table.column_count for _ in range(table.row_count)]
        for cell in table.cells:
            data[cell.row_index][cell.column_index] = cell.content.replace("\n", " ")

        df = pd.DataFrame(data)
        df.dropna(how="all", axis=0, inplace=True)  # Remove empty rows
        df.dropna(how="all", axis=1, inplace=True)  # Remove empty columns

        tables.append(df.to_markdown(index=False))

    return tables, first_positions, last_positions


def extract_content(result, page):
    """Extracts structured content from a document page in Markdown format.
    
    Args:
        result: The document analysis result.
        page: The specific page to process.
    
    Returns:
        str: Extracted content in Markdown format.
    """
    tables, first_positions, last_positions = extract_tables_markdown(result, page)
    
    if not result.paragraphs:
        return ""  # Early exit if no paragraphs are found
    
    markdown_text = ""
    has_text = False
    is_table = False
    table_to_add = ""
    f_p, l_p = -1, -1

    for paragraph_index, paragraph in enumerate(result.paragraphs):
        if paragraph.bounding_regions[0].page_number != page.page_number:
            continue  # Skip paragraphs not on the current page

        if paragraph.role in ["table", "figure"]:
            continue  # Skip non-text content
        
        # Check if current paragraph belongs to a table
        if not is_table and tables:
            for t, f, l in zip(tables, first_positions, last_positions):
                if f <= paragraph_index <= l:
                    is_table, table_to_add, f_p, l_p = True, "### BEGIN TABLE \n\n"+t+"\n\n### END TABLE \n\n", f, l
                    has_text = True
                    break

        if not is_table:
            has_text = True
            if paragraph.role == "pageHeader":
                markdown_text += f"## {paragraph.content}\n\n"
            elif paragraph.role == "sectionHeading":
                markdown_text += f"### {paragraph.content}\n\n"
            else:
                markdown_text += f"{paragraph.content}\n\n"

        if paragraph_index == l_p:
            markdown_text += table_to_add + "\n\n"
            is_table, table_to_add, f_p, l_p = False, "", -1, -1  # Reset table tracking

    return markdown_text.strip() if has_text else ""






def pdf_to_json_azure(path_to_sample_documents, output_json_file_name, azure_key, azure_endpoint):
    document_intelligence_client = DocumentIntelligenceClient(
        endpoint=azure_endpoint,
        credential=AzureKeyCredential(azure_key)
    )

    with open(path_to_sample_documents, "rb") as f:
        poller = document_intelligence_client.begin_analyze_document(
            "prebuilt-layout",
            body=f,
            output_content_format="markdown"
        )
        result: AnalyzeResult = poller.result()

    content = {
        "pages": []
    }

    for page in result.pages:
        text_content = extract_content(result, page)
        content["pages"].append({
            "page_number": page.page_number,
            "content": text_content.strip()
        })

    with open(output_json_file_name, "w", encoding="utf-8") as f:
        json.dump(content, f, indent=2, ensure_ascii=False)


##################### EXTRACT Tables and graphics as images, Then Use LVM ##################################################""

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Fonction pour obtenir les éléments dans un pdf 
def extract_unstructured_elements_from_pdf(path) :
    elements = partition_pdf(
        filename=path,                                         # mandatory
        strategy="hi_res",                                     # mandatory to use ``hi_res`` strategy
        extract_images_in_pdf=True,                            # mandatory to set as ``True``
        extract_image_block_types=["Image", "Table"],          # optional
        extract_image_block_to_payload=False,                  # optional
        extract_image_block_output_dir="00-data/image_outputs",  # optional - only works when ``extract_image_block_to_payload=False``
        )
    return elements

# Fonction pour obtenir le texte à partir des éléments de la partition unstructured
def pdf_to_json_LLM_1(path_to_sample_documents, output_json_file_name, prompt):
    elements = extract_unstructured_elements_from_pdf(path_to_sample_documents)

    content = {
        "pages": []
    }

    page_buffer = {}
    current_page = 1

    for el in elements:
        meta = el.metadata.to_dict()
        page_number = meta.get("page_number", current_page)
        text = ""

        if 'image_path' in meta:
            try:
                path_image = meta['image_path'].replace("\\", "/")
                openai_client = OpenAI()
                base64_image = encode_image(path_image)
                response = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}",
                                        "detail": "high"
                                    },
                                },
                            ],
                        }
                    ],
                    max_tokens=1500,
                )
                text = response.choices[0].message.content or ""

                if text != "":
                    text = "\n### BEGIN TABLE \n\n" + text + "\n\n### END TABLE \n\n"
            except OpenAIError as e:
                print(f"Image processing error on page {page_number}: {e}")


            
        else:
            text = el.text or ""


        page_buffer.setdefault(page_number, []).append(text)

    for page_num in sorted(page_buffer.keys()):
        page_text = "\n".join(page_buffer[page_num])
        content["pages"].append({
            "page_number": page_num,
            "content": page_text.strip()
        })

    with open(output_json_file_name, "w", encoding="utf-8") as f:
        json.dump(content, f, indent=2, ensure_ascii=False)










def encode_image_to_data_url(image_path: str, image_format="jpeg") -> str:
    with open(image_path, "rb") as img_file:
        image_data = base64.b64encode(img_file.read()).decode("utf-8")
    return f"data:image/{image_format};base64,{image_data}"



def pdf_to_json_LLM_11(
    path_to_sample_documents: str,
    output_json_file_name: str,
    prompt: str,
    use_azure: bool = False,
    azure_endpoint: str = "",
    azure_api_key: str = "",
    azure_model_name: str = "gpt-4o"
):
    elements = extract_unstructured_elements_from_pdf(path_to_sample_documents)

    content = {
        "pages": []
    }

    page_buffer = {}
    current_page = 1

    if use_azure:
        client = ChatCompletionsClient(
            endpoint=azure_endpoint,
            credential=AzureKeyCredential(azure_api_key)
        )
    else:
        openai_client = OpenAI()

    for el in elements:
        meta = el.metadata.to_dict()
        page_number = meta.get("page_number", current_page)
        text = ""

        if 'image_path' in meta:
            try:
                image_path = meta['image_path'].replace("\\", "/")
                data_url = encode_image_to_data_url(image_path)

                if use_azure:
                    response = client.complete(
                        model=azure_model_name,
                        messages=[
                            SystemMessage("You are a helpful assistant that can generate responses based on images."),
                            UserMessage(content=[
                                TextContentItem(text=prompt),
                                ImageContentItem(image_url=ImageUrl(url=data_url))
                            ])
                        ],
                        temperature=0.0
                    )
                    text = response.choices[0].message.content or ""

                else:
                    base64_image = data_url.split(",")[1]
                    response = openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{base64_image}",
                                            "detail": "high"
                                        },
                                    },
                                ],
                            }
                        ],

                    )
                    text = response.choices[0].message.content or ""

                if text:
                    text = " \n### BEGIN TABLE \n\n " + text + " \n\n### END TABLE \n\n "

            except Exception as e:
                print(f"[ERROR] Image processing failed on page {page_number}: {e}")
        else:
            text = el.text or ""

        page_buffer.setdefault(page_number, []).append(text)

    for page_num in sorted(page_buffer.keys()):
        page_text = "\n".join(page_buffer[page_num])
        content["pages"].append({
            "page_number": page_num,
            "content": page_text.strip()
        })

    with open(output_json_file_name, "w", encoding="utf-8") as f:
        json.dump(content, f, indent=2, ensure_ascii=False)
        print(f"✅ JSON saved to {output_json_file_name}")












##################### EXTRACT each page as as images, Then Use LVM ##################################################""


    # --- Helper function (same as before) ---
def encode_image_to_base64(image: Image.Image, format="PNG") -> str:
    """Encodes a PIL Image object to a base64 string."""
    buffered = io.BytesIO()
    if format.upper() == "JPEG" and image.mode == "RGBA":
        image = image.convert("RGB")
    image.save(buffered, format=format)
    img_byte = buffered.getvalue()
    base64_str = base64.b64encode(img_byte).decode('utf-8')
    return base64_str


def extract_text_from_pdf_gpt4o_json(
    pdf_path: str,
    json_output_path: str,
    prompt: str,
    image_format: str = "PNG",
    image_detail: str = "high",
    max_tokens_per_page: int = 4000,
    verbose: bool = False
) -> None:
    """
    Extracts structured Markdown from PDF using GPT-4o Vision and saves it as JSON with page-level detail.
    
    Args:
        pdf_path (str): Path to the input PDF.
        json_output_path (str): Path to save the JSON file.
        prompt (str): Prompt to guide GPT-4o.
        ... (other args same as before)
    """
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        return

    try:
        pdfinfo = pdfinfo_from_path(pdf_path)
        num_pages = pdfinfo["Pages"]
        if verbose:
            print(f"PDF detected with {num_pages} pages.")
    except Exception as e:
        print(f"Error reading PDF info: {e}")
        return

    try:
        images = convert_from_path(pdf_path, fmt=image_format.lower())
        if verbose:
            print(f"Converted {len(images)} pages to images.")
    except Exception as e:
        print(f"Image conversion failed: {e}")
        return

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment.")
        return

    client = OpenAI(api_key=api_key)
    page_data = []

    for i, image in enumerate(images):
        page_num = i + 1
        if verbose:
            print(f"Processing page {page_num}...")

        try:
            base64_image = encode_image_to_base64(image, format=image_format)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{image_format.lower()};base64,{base64_image}",
                                "detail": image_detail
                            },
                        },
                    ],
                }
            ]

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=max_tokens_per_page,
            )

            content = response.choices[0].message.content.strip() if response.choices else ""

            page_data.append({
                "page_number": page_num,
                "content": content
            })

        except Exception as e:
            print(f"Error on page {page_num}: {e}")
            page_data.append({
                "page_number": page_num,
                "content": ""
            })

    # Save to JSON
    try:
        os.makedirs(os.path.dirname(json_output_path), exist_ok=True)
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(page_data, f, ensure_ascii=False, indent=2)
        print(f"Successfully saved output to: {json_output_path}")
    except Exception as e:
        print(f"Failed to write JSON file: {e}")


def extract_text_from_pdf_azure_vision_json(
    pdf_path: str,
    json_output_path: str,
    prompt: str,
    azure_endpoint: str,
    azure_api_key: str,
    azure_model_name: str = "gpt-4o",
    image_format: str = "PNG",
    max_tokens_per_page: int = 2048,
    verbose: bool = False
) -> None:
    """
    Extracts structured Markdown from PDF using Azure GPT-4o Vision and saves it as JSON with page-level detail.
    """
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return

    try:
        pdfinfo = pdfinfo_from_path(pdf_path)
        num_pages = pdfinfo["Pages"]
        if verbose:
            print(f"📄 PDF has {num_pages} pages.")
    except Exception as e:
        print(f"❌ Failed to read PDF metadata: {e}")
        return

    try:
        images = convert_from_path(pdf_path, fmt=image_format.lower())
        if verbose:
            print(f"🖼️ Converted {len(images)} pages to images.")
    except Exception as e:
        print(f"❌ Image conversion failed: {e}")
        return

    client = ChatCompletionsClient(
        endpoint=azure_endpoint,
        credential=AzureKeyCredential(azure_api_key)
    )

    page_data = []

    for i, image in enumerate(images):
        page_num = i + 1
        if verbose:
            print(f"🔍 Processing page {page_num}...")

        try:
            base64_image = encode_image_to_base64(image, format=image_format)
            data_url = f"data:image/{image_format.lower()};base64,{base64_image}"

            response = client.complete(
                model=azure_model_name,
                messages=[
                    SystemMessage("You are a helpful assistant that extracts structured data from PDF images."),
                    UserMessage(content=[
                        TextContentItem(text=prompt),
                        ImageContentItem(image_url=ImageUrl(url=data_url))
                    ])
                ],
                max_tokens=max_tokens_per_page,
                temperature=0.0
            )

            content = response.choices[0].message.content.strip() if response.choices else ""

            if verbose:
                print(f"✅ Page {page_num} processed.")

            page_data.append({
                "page_number": page_num,
                "content": content
            })

        except Exception as e:
            print(f"⚠️ Error on page {page_num}: {e}")
            page_data.append({
                "page_number": page_num,
                "content": ""
            })

    # Save JSON output
    try:
        os.makedirs(os.path.dirname(json_output_path), exist_ok=True)
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(page_data, f, ensure_ascii=False, indent=2)
        print(f"💾 Output saved to: {json_output_path}")
    except Exception as e:
        print(f"❌ Failed to write JSON file: {e}")