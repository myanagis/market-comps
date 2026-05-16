import logging
from prefect import flow
from market_comps.document_pipeline.tasks_transcription import transcribe_document
from market_comps.document_pipeline.tasks_classification import classify_document
from market_comps.document_pipeline.tasks_extraction import extract_data
from market_comps.pdf_parser.models import ParserResult

logger = logging.getLogger(__name__)

@flow(name="process_document_pipeline")
def process_document_pipeline(file_bytes: bytes, filename: str, extraction_method: str, model: str) -> ParserResult:
    """
    Main Prefect flow that orchestrates the document parsing process:
    1. Transcribe
    2. Classify
    3. Extract/Summarize
    """
    logger.info(f"Starting document pipeline for {filename} using {extraction_method} method and {model} model.")
    
    # 1. Transcribe
    transcription_text, transcribe_usage = transcribe_document(
        pdf_bytes=file_bytes, 
        filename=filename, 
        method=extraction_method, 
        model=model
    )
    
    # 2. Classify
    classification, classify_usage = classify_document(
        document_text=transcription_text, 
        model=model
    )
    
    # 3. Extract
    result, extract_usage = extract_data(
        document_text=transcription_text, 
        classification=classification, 
        model=model,
        filename=filename
    )
    
    # Consolidate usage and metadata
    result.pdf_engine = extraction_method
    
    total_usage = result.llm_usage
    for u in [transcribe_usage, classify_usage, extract_usage]:
        if u:
            total_usage.total_prompt_tokens += u.total_prompt_tokens
            total_usage.total_completion_tokens += u.total_completion_tokens
            total_usage.total_tokens += u.total_tokens
            total_usage.estimated_cost_usd += u.estimated_cost_usd
            total_usage.call_count += u.call_count

    return result
