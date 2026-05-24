import os
import sys
import datetime
import json
import hashlib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from market_comps.db.session import get_db
from market_comps.db.models import IngestionRun, SourceDocument, DocumentText, ExtractionJob
from market_comps.ingestion.classifier import classify_document, get_recommended_schemas
from market_comps.llm_client import LLMClient
from market_comps.ingestion.extractor import extract_entities_from_text

text_content = """
This is a startup pitch deck for Acme Corp.
We are a B2B SaaS company that helps restaurants manage inventory.
We are raising $5M in a Seed round.
Our team consists of former Google and Facebook engineers.
"""

db = next(get_db())

try:
    content_hash = hashlib.sha256(text_content.encode()).hexdigest()
    
    from market_comps.ingestion.async_processor import start_document_ingestion
    import time
    
    run_id = start_document_ingestion(
        db=db,
        text_content=text_content,
        file_name="test_pipeline.txt",
        storage_path=None,
        content_hash=content_hash,
        final_instructions=""
    )
    
    print(f"Started ingestion run {run_id}. Waiting for completion...")
    
    while True:
        run = db.query(IngestionRun).get(run_id)
        db.refresh(run)
        if run.run_status != "RUNNING":
            print(f"Run finished with status: {run.run_status}")
            if run.error_message:
                print(f"Error: {run.error_message}")
            break
        time.sleep(2)
        
    print("Test Complete.")
    
except Exception as e:
    db.rollback()
    import traceback
    traceback.print_exc()
