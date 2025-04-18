import argparse

import os
import io
import json
import base64
from typing import List, Optional

from PIL import Image
from pdf2image import convert_from_path
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import ElementMetadata
from unstructured_inference.models.base import BaseInferenceEngine
from openai import OpenAI, OpenAIError

# For Azure
from azure.core.credentials import AzureKeyCredential
from openai.types import (
    ChatCompletionMessage,
)
from openai import AzureOpenAI as ChatCompletionsClient
from openai.types.chat import (
    ImageContentItem,
    TextContentItem,
    UserMessage,
    SystemMessage,
    ImageUrl,
)


class PDFToTextExtractor:
    def __init__(
        self,
        use_azure: bool = False,
        azure_endpoint: Optional[str] = None,
        azure_api_key: Optional[str] = None,
        azure_model_name: str = "gpt-4o",
    ):
        self.use_azure = use_azure
        self.azure_model_name = azure_model_name

        if self.use_azure:
            self.client = ChatCompletionsClient(
                endpoint=azure_endpoint,
                credential=AzureKeyCredential(azure_api_key),
            )
        else:
            self.client = OpenAI()

    @staticmethod
    def encode_image_to_data_url(image_path: str, image_format="jpeg") -> str:
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/{image_format};base64,{image_data}"

    @staticmethod
    def encode_pil_image_to_base64(image: Image.Image, image_format="PNG") -> str:
        buffered = io.BytesIO()
        if image_format.upper() == "JPEG" and image.mode == "RGBA":
            image = image.convert("RGB")
        image.save(buffered, format=image_format)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _call_llm_with_image(self, base64_data_url: str, prompt: str) -> str:
        try:
            if self.use_azure:
                response = self.client.complete(
                    model=self.azure_model_name,
                    messages=[
                        SystemMessage("You are a helpful assistant."),
                        UserMessage(content=[
                            TextContentItem(text=prompt),
                            ImageContentItem(image_url=ImageUrl(url=base64_data_url)),
                        ]),
                    ],
                    temperature=0.0,
                )
                return response.choices[0].message.content or ""
            else:
                base64_image = base64_data_url.split(",")[1]
                response = self.client.chat.completions.create(
                    model=self.azure_model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}",
                                        "detail": "high",
                                    },
                                },
                            ],
                        }
                    ],
                )
                return response.choices[0].message.content or ""
        except Exception as e:
            print(f"[LLM ERROR] {e}")
            return ""

    def extract_from_partitioned_elements(
        self,
        pdf_path: str,
        output_json_path: str,
        prompt: str,
        extract_dir: str = "00-data/image_outputs"
    ):
        elements = partition_pdf(
            filename=pdf_path,
            strategy="hi_res",
            extract_images_in_pdf=True,
            extract_image_block_types=["Image", "Table"],
            extract_image_block_output_dir=extract_dir,
            extract_image_block_to_payload=False,
        )

        page_buffer = {}
        current_page = 1

        for el in elements:
            meta = el.metadata.to_dict()
            page_number = meta.get("page_number", current_page)
            text = ""

            if 'image_path' in meta:
                image_path = meta['image_path'].replace("\\", "/")
                data_url = self.encode_image_to_data_url(image_path)
                text = self._call_llm_with_image(data_url, prompt)
                if text:
                    text = "\n### BEGIN TABLE\n" + text.strip() + "\n### END TABLE\n"
            else:
                text = el.text or ""

            page_buffer.setdefault(page_number, []).append(text)

        content = {
            "pages": [
                {"page_number": pg, "content": "\n".join(txts).strip()}
                for pg, txts in sorted(page_buffer.items())
            ]
        }

        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(content, f, indent=2, ensure_ascii=False)
        print(f"✅ JSON saved: {output_json_path}")

    def extract_from_page_images(
        self,
        pdf_path: str,
        output_json_path: str,
        prompt: str,
        image_format: str = "PNG",
        max_tokens_per_page: int = 2048,
        verbose: bool = False,
    ):
        try:
            images = convert_from_path(pdf_path, fmt=image_format.lower())
        except Exception as e:
            print(f"❌ Image conversion failed: {e}")
            return

        content = {"pages": []}

        for i, img in enumerate(images):
            page_number = i + 1
            if verbose:
                print(f"🔍 Processing page {page_number}")

            base64_image = self.encode_pil_image_to_base64(img, image_format)
            data_url = f"data:image/{image_format.lower()};base64,{base64_image}"
            text = self._call_llm_with_image(data_url, prompt)
            content["pages"].append({
                "page_number": page_number,
                "content": text.strip()
            })

        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(content, f, indent=2, ensure_ascii=False)
        print(f"✅ Output saved: {output_json_path}")





def main():
    parser = argparse.ArgumentParser(description="Extract text from PDF using LLMs.")
    parser.add_argument("--pdf", type=str, required=True, help="Path to the input PDF file.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output JSON file.")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to send to the LLM.")
    parser.add_argument("--strategy", choices=["partition", "page"], default="partition",
                        help="Extraction strategy: 'partition' for element-level or 'page' for full-page images.")
    parser.add_argument("--use_azure", action="store_true", help="Use Azure OpenAI instead of OpenAI.")
    parser.add_argument("--azure_endpoint", type=str, default="", help="Azure OpenAI endpoint.")
    parser.add_argument("--azure_key", type=str, default="", help="Azure OpenAI API key.")
    parser.add_argument("--verbose", action="store_true", help="Print progress logs.")

    args = parser.parse_args()

    extractor = PDFToTextExtractor(
        use_azure=args.use_azure,
        azure_endpoint=args.azure_endpoint,
        azure_api_key=args.azure_key,
    )

    if args.strategy == "partition":
        extractor.extract_from_partitioned_elements(
            pdf_path=args.pdf,
            output_json_path=args.output,
            prompt=args.prompt,
        )
    else:
        extractor.extract_from_page_images(
            pdf_path=args.pdf,
            output_json_path=args.output,
            prompt=args.prompt,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()