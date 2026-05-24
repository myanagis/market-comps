import threading
import logging
from datetime import datetime
import json
import traceback

from market_comps.db.session import get_db
from market_comps.db.models import IngestionRun, SourceDocument, DocumentText, ExtractionJob
from market_comps.ingestion.classifier import classify_document, get_recommended_schemas
from market_comps.llm_client import LLMClient
from market_comps.ingestion.extractor import extract_entities_from_text
from market_comps.ingestion.reconciler import reconcile_all
from market_comps.models import LLMUsage

logger = logging.getLogger(__name__)

def run_document_ingestion_background(run_id: int, text_content: str, final_instructions: str, transcription_usage=None):
    """
    Executes the heavy lifting (classification, extraction, reconciliation) in a background thread.
    """
    db = next(get_db())
    
    try:
        run = db.query(IngestionRun).get(run_id)
        if not run:
            logger.error(f"IngestionRun {run_id} not found in background thread.")
            return

        source_doc = db.query(SourceDocument).filter_by(ingestion_run_id=run.id).first()
        doc_text = db.query(DocumentText).filter_by(source_document_id=source_doc.id).first()

        # 3.5 Classify Document
        logger.info(f"Classifying document for run {run.id}...")
        try:
            class_result, class_usage = classify_document(text_content, LLMClient())
            doc_class = class_result.get("document_type", "unknown")
            source_doc.document_class = doc_class
            source_doc.classification_result_json = class_result
            db.flush()
            
            schemas_to_run = get_recommended_schemas(doc_class)
            if not schemas_to_run:
                schemas_to_run = ["DOCUMENT_ENTITIES"]
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            source_doc.document_class = "unknown"
            source_doc.classification_result_json = {"error": str(e), "traceback": traceback.format_exc()}
            db.flush()
            schemas_to_run = ["DOCUMENT_ENTITIES"]
            class_usage = None
        
        # 4. Run LLM Extraction
        total_entities_extracted = 0
        schema_name = schemas_to_run[0] if schemas_to_run else "DOCUMENT_ENTITIES"
        
        logger.info(f"Running extraction schema: {schema_name}...")
        
        job = ExtractionJob(
            ingestion_run_id=run.id,
            document_text_id=doc_text.id,
            schema_name=schema_name,
            status="IN_PROGRESS",
            started_at=datetime.utcnow()
        )
        db.add(job)
        db.flush()
        
        try:
            extraction_result = extract_entities_from_text(
                db, run, job, doc_text, schema_name, {"llm_instruction": final_instructions}
            )
            
            total_entities_extracted += extraction_result["entities_extracted"]
            job_usage = extraction_result["llm_usage"]
            
            usage_obj = LLMUsage(**job_usage)
            if transcription_usage:
                usage_obj.total_prompt_tokens += transcription_usage.total_prompt_tokens
                usage_obj.total_completion_tokens += transcription_usage.total_completion_tokens
                usage_obj.total_tokens += transcription_usage.total_tokens
                usage_obj.estimated_cost_usd += transcription_usage.estimated_cost_usd
                usage_obj.call_count += transcription_usage.call_count
                usage_obj.traces = transcription_usage.traces + usage_obj.traces
            if class_usage:
                usage_obj.total_prompt_tokens += class_usage.total_prompt_tokens
                usage_obj.total_completion_tokens += class_usage.total_completion_tokens
                usage_obj.total_tokens += class_usage.total_tokens
                usage_obj.estimated_cost_usd += class_usage.estimated_cost_usd
                usage_obj.call_count += class_usage.call_count
                usage_obj.traces = class_usage.traces + usage_obj.traces
                
            job_usage = usage_obj.model_dump(mode="json")
            job.llm_usage_json = json.loads(json.dumps(job_usage, default=str))
            job.status = "SUCCESS"
            job.completed_at = datetime.utcnow()
            db.flush()
        except Exception as e:
            job.status = "FAILED"
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            db.flush()
            raise e # Bubble up to the main run exception handler
        
        # 5. Reconcile
        logger.info("Reconciling combined results against CRM...")
        reconcile_stats = reconcile_all(db, run, None)
        
        # 6. Complete
        total_created = reconcile_stats.get("orgs_reconciled", 0) + reconcile_stats.get("people_reconciled", 0)
        run.run_status = "SUCCESS"
        run.completed_at = datetime.utcnow()
        run.records_processed = total_entities_extracted
        run.records_created = total_created
        
        db.commit()
        logger.info(f"Run {run.id} completed successfully.")
        
    except Exception as e:
        db.rollback()
        logger.error(f"Pipeline {run_id} failed: {e}")
        try:
            # Re-fetch the run in a new transaction to save the error state
            db.commit()
            run_failed = db.query(IngestionRun).get(run_id)
            if run_failed:
                run_failed.run_status = "FAILED"
                run_failed.error_message = f"{str(e)}\n\n{traceback.format_exc()}"
                run_failed.completed_at = datetime.utcnow()
                db.commit()
        except Exception as inner_e:
            logger.error(f"Failed to log error to DB for run {run_id}: {inner_e}")
    finally:
        db.close()


def start_document_ingestion(db, text_content: str, file_name: str, storage_path: str, content_hash: str, final_instructions: str, transcription_usage=None) -> int:
    """
    Synchronously creates the IngestionRun and SourceDocument records, then spins up a background thread.
    Returns the IngestionRun ID.
    """
    run = IngestionRun(
        pipeline_id=None,
        run_status="RUNNING",
        started_at=datetime.utcnow()
    )
    db.add(run)
    db.flush()
    
    source_doc = SourceDocument(
        ingestion_run_id=run.id,
        document_type="PDF" if file_name.lower().endswith(".pdf") else "TEXT",
        source_url=file_name,
        file_path=storage_path,
        content_hash=content_hash,
    )
    db.add(source_doc)
    db.flush()
    
    doc_text = DocumentText(
        source_document_id=source_doc.id,
        data_type="DOCUMENT_TEXT",
        raw_content=text_content,
        content_hash=content_hash,
    )
    db.add(doc_text)
    db.flush()
    db.commit()
    
    run_id = run.id
    
    # Start background thread
    thread = threading.Thread(
        target=run_document_ingestion_background,
        args=(run_id, text_content, final_instructions, transcription_usage),
        daemon=True
    )
    thread.start()
    
    return run_id
