import base64
import logging
from pathlib import Path
from typing import Tuple

import fitz  # PyMuPDF
from prefect import task, flow
from market_comps.llm_client import LLMClient
from market_comps.pdf_parser.pdf_client import PDFClient
from market_comps.models import LLMUsage

logger = logging.getLogger(__name__)

_DIR = Path(__file__).resolve().parent.parent / "pdf_parser"
_INSTRUCTIONS_PATH = _DIR / "vlm_instructions.md"

def _load_instructions() -> str:
    with _INSTRUCTIONS_PATH.open("r", encoding="utf-8") as f:
        return f.read()

@task(name="extract_text_with_vlm")
def extract_text_with_vlm(pdf_bytes: bytes, model: str, instructions: str) -> Tuple[str, LLMUsage]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    messages = [{"role": "user", "content": [{"type": "text", "text": instructions}]}]
    
    for page in doc:
        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("jpeg")
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        messages[0]["content"].append({
            "type": "image_url", 
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })
        
    client = LLMClient(model=model)
    return client.chat_completion(messages=messages, temperature=0.0)

@task(name="extract_text_with_ocr")
def extract_text_with_ocr(pdf_bytes: bytes, filename: str, model: str, instructions: str) -> Tuple[str, LLMUsage]:
    client = PDFClient(pdf_engine="mistral-ocr", model=model)
    content, _, step_usage = client.send(prompt=instructions, pdf_bytes=pdf_bytes, filename=filename, temperature=0.0)
    return content, step_usage

@task(name="extract_text_with_native")
def extract_text_with_native(pdf_bytes: bytes, filename: str, model: str, instructions: str) -> Tuple[str, LLMUsage]:
    if filename.lower().endswith(".pptx"):
        import io
        from pptx import Presentation
        prs = Presentation(io.BytesIO(pdf_bytes))
        text_runs = []
        for slide_num, slide in enumerate(prs.slides):
            slide_text = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    slide_text.append(shape.text.strip())
            if slide_text:
                text_runs.append(f"--- Slide {slide_num + 1} ---\n" + "\n".join(slide_text))
        return "\n\n".join(text_runs), LLMUsage()
    else:
        client = PDFClient(pdf_engine="pdf-text", model=model)
        content, _, step_usage = client.send(prompt=instructions, pdf_bytes=pdf_bytes, filename=filename, temperature=0.0)
        return content, step_usage

@task(name="reconcile_texts")
def reconcile_texts(text_content: str, vlm_content: str, text_usage: LLMUsage, vlm_usage: LLMUsage, model: str) -> Tuple[str, LLMUsage]:
    merge_prompt = f"""
You are a specialized VC Technical Analyst merging two transcriptions of the same pitch deck.

TEXT EXTRACTION (Highly accurate numbers, poor layout):
{text_content}

VLM EXTRACTION (Great layout, but may hallucinate numbers):
{vlm_content}

INSTRUCTIONS:
1. Produce a final, unified Markdown transcription.
2. Use the VLM version for the overall structure, tables, and graphs.
3. STRICTLY cross-reference every number, financial metric, and scientific unit against the TEXT EXTRACTION. 
4. If the VLM hallucinated a number, replace it with the correct number from the TEXT EXTRACTION.
5. Output ONLY the final markdown. Do not include your reasoning or introductory text.
"""
    
    client = LLMClient(model=model)
    final_content, final_usage = client.simple_text(prompt=merge_prompt, temperature=0.0)
    
    # Merge all usages
    total_usage = LLMUsage()
    for u in [text_usage, vlm_usage, final_usage]:
        total_usage.total_prompt_tokens += u.total_prompt_tokens
        total_usage.total_completion_tokens += u.total_completion_tokens
        total_usage.total_tokens += u.total_tokens
        total_usage.estimated_cost_usd += u.estimated_cost_usd
        total_usage.call_count += u.call_count
        
    return final_content, total_usage

@flow(name="transcribe_document")
def transcribe_document(pdf_bytes: bytes, filename: str, method: str, model: str) -> Tuple[str, LLMUsage]:
    """
    Process a PDF document using one of four methods:
    'ocr', 'text', 'vlm', or 'vlm_plus_text'.
    Returns (markdown_text, LLMUsage)
    """
    instructions = _load_instructions()
    
    if method == "ocr":
        return extract_text_with_ocr(pdf_bytes, filename, model, instructions)
        
    elif method == "text":
        return extract_text_with_native(pdf_bytes, filename, model, instructions)
        
    elif method == "vlm":
        return extract_text_with_vlm(pdf_bytes, model, instructions)
        
    elif method == "vlm_plus_text":
        logger.info("Running VLM_PLUS_TEXT hybrid extraction...")
        # Run text extraction
        text_content, text_usage = extract_text_with_native(pdf_bytes, filename, model, instructions)
        # Run VLM extraction
        vlm_content, vlm_usage = extract_text_with_vlm(pdf_bytes, model, instructions)
        # Cross compare
        return reconcile_texts(text_content, vlm_content, text_usage, vlm_usage, model)
        
    else:
        raise ValueError(f"Unknown method: {method}")
