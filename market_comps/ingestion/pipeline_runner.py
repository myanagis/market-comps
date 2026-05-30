"""
Pipeline Runner — Orchestrator
================================
Top-level entry point that coordinates the ETL steps.

    Step 1: Create PipelineRun record
    Step 2: EXTRACT — Fetch raw content → save to extracted_data_raw
    Step 3: TRANSFORM — LLM extraction → save to extracted_entities + extracted_relationships
    Step 4: LOAD — Reconcile against CRM → write to organizations/people/etc. + entity_audit_trail
    Step 5: Complete PipelineRun
"""

import hashlib
import logging
from datetime import datetime
from urllib.parse import urljoin

from sqlalchemy.orm import Session

from market_comps.db.models import Pipeline, PipelineRun, PipelineRunStep, SourceDocument, DocumentText, ExtractionJob, ExtractedEntity, ExtractedRelationship
from market_comps.ingestion.scraper import fetch_page_text
from market_comps.ingestion.extractor import extract_entities_from_text, extract_profile_detail
from market_comps.ingestion.reconciler import reconcile_all

logger = logging.getLogger(__name__)


def run_pipeline(db: Session, pipeline_id: int) -> PipelineRun:
    """Main entry point: runs a full pipeline.

    Steps:
        1. Create PipelineRun record
        2. EXTRACT — Fetch raw content → ExtractedDataRaw
        3. TRANSFORM — LLM extraction → ExtractedEntity + ExtractedRelationship
        4. LOAD — Reconcile against CRM → Organization/Person/etc. + EntityAuditTrail
        5. Complete PipelineRun
    """
    # --- STEP 1: Load pipeline, create run ---
    pipeline = db.query(Pipeline).filter(Pipeline.id == pipeline_id).first()
    if not pipeline:
        raise ValueError(f"Pipeline {pipeline_id} not found")

    config = pipeline.config_json or {}

    run = PipelineRun(
        pipeline_id=pipeline.id,
        run_status="RUNNING",
        started_at=datetime.utcnow(),
        created_at=datetime.utcnow()
    )
    db.add(run)
    db.commit()
    try:
        from market_comps.ingestion.registry import get_fetcher, get_preparer, get_extractor, get_normalizer, get_updater

        config = pipeline.config_json or {}

        # 1. FETCH
        logger.info(f"[Pipeline {pipeline.id}] Step 1: Fetching (connector={pipeline.connector_type})")
        fetch_step = PipelineRunStep(
            pipeline_run_id=run.id, step_order=1, step_name="FETCH",
            step_type="FETCH", method=pipeline.connector_type,
            started_at=datetime.utcnow(), status="RUNNING"
        )
        db.add(fetch_step)
        db.flush()

        fetcher = get_fetcher(pipeline.connector_type)
        raw_data = fetcher.fetch_data(db, run, pipeline)

        fetch_step.completed_at = datetime.utcnow()
        fetch_step.status = "SUCCESS"
        db.add(fetch_step)
        db.flush()

        # 2. PREPARE
        logger.info(f"[Pipeline {pipeline.id}] Step 2: Prepare (parser={pipeline.parser_type})")
        prep_step = PipelineRunStep(
            pipeline_run_id=run.id, step_order=2, step_name="PREPARE",
            step_type="PREPARE", method=pipeline.parser_type,
            started_at=datetime.utcnow(), status="RUNNING"
        )
        db.add(prep_step)
        db.flush()

        preparer = get_preparer(pipeline.parser_type)
        prepared_data = preparer.prepare_raw_data(db, run, pipeline, raw_data)

        prep_step.completed_at = datetime.utcnow()
        prep_step.status = "SUCCESS"
        db.add(prep_step)
        db.flush()

        # 3. EXTRACT
        logger.info(f"[Pipeline {pipeline.id}] Step 3: Extract (parser={pipeline.parser_type})")
        ext_step = PipelineRunStep(
            pipeline_run_id=run.id, step_order=3, step_name="EXTRACT",
            step_type="ATTRIBUTE_EXTRACTION", method=pipeline.parser_type,
            started_at=datetime.utcnow(), status="RUNNING"
        )
        db.add(ext_step)
        db.flush()

        extractor = get_extractor(pipeline.parser_type)
        extracted_data = extractor.extract_attributes(db, run, pipeline, prepared_data)

        ext_step.completed_at = datetime.utcnow()
        ext_step.status = "SUCCESS"
        db.add(ext_step)
        db.flush()

        # 4. NORMALIZE
        logger.info(f"[Pipeline {pipeline.id}] Step 4: Normalize (normalizer={pipeline.normalizer_type})")
        norm_step = PipelineRunStep(
            pipeline_run_id=run.id, step_order=4, step_name="NORMALIZE",
            step_type="NORMALIZATION", method=pipeline.normalizer_type,
            started_at=datetime.utcnow(), status="RUNNING"
        )
        db.add(norm_step)
        db.flush()

        normalizer = get_normalizer(pipeline.normalizer_type)
        normalized_data = normalizer.normalize_data(db, run, pipeline, extracted_data)

        norm_step.completed_at = datetime.utcnow()
        norm_step.status = "SUCCESS"
        db.add(norm_step)
        db.flush()

        # 5. UPDATE
        # Note: the generic updater handles reconciliation logic currently.
        # We assume the config specifies updater type if we decouple it entirely in the future.
        updater_type = config.get("updater_type", "RECORD_UPDATER")
        logger.info(f"[Pipeline {pipeline.id}] Step 5: Update (updater={updater_type})")
        upd_step = PipelineRunStep(
            pipeline_run_id=run.id, step_order=5, step_name="UPDATE",
            step_type="UPDATE", method=updater_type,
            started_at=datetime.utcnow(), status="RUNNING"
        )
        db.add(upd_step)
        db.flush()

        updater = get_updater(updater_type)
        update_stats = updater.update_records(db, run, pipeline, normalized_data)

        upd_step.completed_at = datetime.utcnow()
        upd_step.status = "SUCCESS"
        db.add(upd_step)
        db.flush()

        all_logs = {"steps": []}
        all_logs["steps"].append({"reconciliation": update_stats})
        if isinstance(extracted_data, dict) and "deep_logs" in extracted_data:
            all_logs["steps"].append({"step": "deep_scrape", "profiles": extracted_data["deep_logs"]})

        # --- STEP 6: Complete ---
        total_created = update_stats.get("orgs_created", 0) + update_stats.get("people_created", 0) + update_stats.get("funds_created", 0)
        run.run_status = "SUCCESS"
        run.completed_at = datetime.utcnow()
        run.records_processed = 0 # Placeholder for total processed records
        run.records_created = total_created
        run.logs_json = all_logs

        pipeline.last_run_at = datetime.utcnow()
        pipeline.last_success_at = datetime.utcnow()

        db.commit()

    except Exception as e:
        logger.error(f"Pipeline run failed: {e}", exc_info=True)
        db.rollback()

        failed_run = db.query(PipelineRun).filter(PipelineRun.id == run.id).first()
        if failed_run:
            failed_run.run_status = "FAILED"
            failed_run.error_message = str(e)
            failed_run.completed_at = datetime.utcnow()
            db.commit()

    return run


def _merge_profile_into_entity(db: Session, run: PipelineRun, profile_job: ExtractionJob, company_name: str, detail: dict):
    """Merge deep-scraped profile detail into the matching ExtractedEntity's payload.
    
    Also creates ExtractedEntity (PERSON) and ExtractedRelationship (FOUNDER_OF)
    records for any founders discovered in the profile detail.
    """

    if not detail or not company_name:
        return

    entity = db.query(ExtractedEntity).join(ExtractionJob).filter(
        ExtractionJob.pipeline_run_id == run.id,
        ExtractedEntity.entity_type == "ORGANIZATION",
        ExtractedEntity.normalized_name == company_name.lower()
    ).first()

    if not entity:
        return

    payload = dict(entity.extracted_payload_json or {})
    
    # Map common LLM variations to our schema keys
    key_map = {
        "company_website": "url",
        "website": "url",
        "linkedin": "linkedin_url",
        "linkedin_url": "linkedin_url",
        "description": "description",
        "industry": "industry",
        "founded_year": "founded_year",
        "founders": "founders"
    }
    
    for src_key, target_key in key_map.items():
        if detail.get(src_key):
            payload[target_key] = detail[src_key]

    entity.extracted_payload_json = payload
    db.flush()
    logger.info(f"[Pipeline] Merged profile detail into {company_name}. Payload keys: {list(payload.keys())}")
    logger.info(f"[Pipeline] Detail object keys: {list(detail.keys())}")
    if "founders" in detail:
        logger.info(f"[Pipeline] Found {len(detail['founders'])} founders in detail for {company_name}")
    else:
        logger.info(f"[Pipeline] NO founders key in detail for {company_name}")

    # Create PERSON entities + FOUNDER_OF relationships for newly discovered founders
    # The 'founders' key is now safely in 'payload' after the mapping above
    founders = payload.get("founders", [])
    for f in founders:
        if not isinstance(f, dict):
            continue
        
        # Handle both "name" (single field) and "first_name"/"last_name" (split) formats
        fname = f.get("first_name", "")
        lname = f.get("last_name", "")
        if not fname and not lname:
            full = f.get("name", "").strip()
            if not full:
                continue
            parts = full.split(None, 1)
            fname = parts[0]
            lname = parts[1] if len(parts) > 1 else ""
            # Normalize the payload so reconciler gets clean data
            f["first_name"] = fname
            f["last_name"] = lname
        if not fname:
            continue

        person_entity = ExtractedEntity(
            extraction_job_id=profile_job.id,
            entity_type="PERSON",
            raw_name=f"{fname} {lname}",
            normalized_name=f"{fname} {lname}".lower(),
            extracted_payload_json=f,
        )
        db.add(person_entity)
        db.flush()
        logger.info(f"[Pipeline] Created ExtractedEntity (PERSON): {fname} {lname} for {company_name}")

        rel = ExtractedRelationship(
            extraction_job_id=profile_job.id,
            relationship_type="FOUNDER_OF",
            source_extracted_entity_id=person_entity.id,
            source_entity_type="PERSON",
            target_extracted_entity_id=entity.id,
            target_entity_type="ORGANIZATION",
            relationship_payload_json={"title": f.get("title", "Founder")},
        )
        db.add(rel)

    db.flush()
