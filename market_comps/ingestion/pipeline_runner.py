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

from market_comps.db.models import Pipeline, PipelineRun, ExtractedDataRaw, ExtractedEntity, ExtractedRelationship
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
    db.refresh(run)

    try:
        url = pipeline.source_url or ""
        is_deep_scrape = config.get("deep_scrape", False)

        # --- STEP 2: EXTRACT — Fetch raw content ---
        logger.info(f"[Pipeline {pipeline.id}] Step 2: Fetching {url}")
        text_content = fetch_page_text(url)

        content_hash = hashlib.sha256(text_content.encode()).hexdigest()
        raw_data = ExtractedDataRaw(
            pipeline_run_id=run.id,
            data_type="PAGE_TEXT",
            source_url=url,
            raw_content=text_content,
            content_hash=content_hash,
            created_at=datetime.utcnow()
        )
        db.add(raw_data)
        db.flush()

        all_logs = {"steps": []}

        # --- STEP 3: TRANSFORM — LLM extraction ---
        logger.info(f"[Pipeline {pipeline.id}] Step 3: LLM extraction (type={pipeline.pipeline_type})")
        extraction_result = extract_entities_from_text(
            db, run, raw_data, pipeline.pipeline_type, config
        )
        all_logs["steps"].append({
            "step": "extract_directory",
            "entities": extraction_result["entities_extracted"],
            "relationships": extraction_result["relationships_extracted"],
            "llm_usage": extraction_result["llm_usage"]
        })

        # Deep scrape: visit profile pages for enrichment
        if is_deep_scrape:
            companies_raw = extraction_result.get("companies_raw", [])
            deep_logs = []

            for i, c in enumerate(companies_raw):
                if not isinstance(c, dict):
                    continue
                profile_path = c.get("profile_path") or c.get("detail_page_path") or ""
                if not profile_path:
                    deep_logs.append({"company": c.get("name"), "status": "no_profile_path"})
                    continue

                profile_url = urljoin(url, profile_path)
                logger.info(f"[Pipeline {pipeline.id}] Deep scrape ({i+1}/{len(companies_raw)}): {profile_url}")

                try:
                    profile_text = fetch_page_text(profile_url)
                    profile_hash = hashlib.sha256(profile_text.encode()).hexdigest()

                    profile_raw = ExtractedDataRaw(
                        pipeline_run_id=run.id,
                        data_type="PROFILE_TEXT",
                        source_url=profile_url,
                        raw_content=profile_text,
                        content_hash=profile_hash,
                        created_at=datetime.utcnow()
                    )
                    db.add(profile_raw)
                    db.flush()

                    detail_result = extract_profile_detail(
                        db, run, profile_raw, c.get("name", "Unknown")
                    )
                    deep_logs.append({
                        "company": c.get("name"),
                        "profile_url": profile_url,
                        "status": "success",
                        "detail": detail_result.get("detail"),
                        "llm_usage": detail_result.get("llm_usage"),
                        "text_preview": profile_text[:1500]
                    })

                    # Merge detail into the matching ExtractedEntity
                    _merge_profile_into_entity(db, run, c.get("name"), detail_result.get("detail", {}))

                except Exception as e:
                    deep_logs.append({
                        "company": c.get("name"),
                        "profile_url": profile_url,
                        "status": "error",
                        "error": str(e)
                    })
                    logger.warning(f"[Pipeline] Deep scrape failed for {profile_url}: {e}")

            all_logs["steps"].append({"step": "deep_scrape", "profiles": deep_logs})

        # --- STEP 4: LOAD — Reconcile against CRM ---
        logger.info(f"[Pipeline {pipeline.id}] Step 4: Reconciling entities against CRM")
        reconcile_stats = reconcile_all(db, run, pipeline)
        all_logs["steps"].append({"step": "reconcile", **reconcile_stats})

        # --- STEP 5: Complete ---
        total_created = reconcile_stats.get("orgs_reconciled", 0) + reconcile_stats.get("people_reconciled", 0)
        run.run_status = "SUCCESS"
        run.completed_at = datetime.utcnow()
        run.records_processed = extraction_result["entities_extracted"]
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


def _merge_profile_into_entity(db: Session, run: PipelineRun, company_name: str, detail: dict):
    """Merge deep-scraped profile detail into the matching ExtractedEntity's payload.
    
    Also creates ExtractedEntity (PERSON) and ExtractedRelationship (FOUNDER_OF)
    records for any founders discovered in the profile detail.
    """

    if not detail or not company_name:
        return

    entity = db.query(ExtractedEntity).filter_by(
        pipeline_run_id=run.id,
        entity_type="ORGANIZATION",
        normalized_name=company_name.lower()
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
            pipeline_run_id=run.id,
            extracted_data_raw_id=entity.extracted_data_raw_id,
            entity_type="PERSON",
            raw_name=f"{fname} {lname}",
            normalized_name=f"{fname} {lname}".lower(),
            extracted_payload_json=f,
            created_at=datetime.utcnow()
        )
        db.add(person_entity)
        db.flush()
        logger.info(f"[Pipeline] Created ExtractedEntity (PERSON): {fname} {lname} for {company_name}")

        rel = ExtractedRelationship(
            pipeline_run_id=run.id,
            extracted_data_raw_id=entity.extracted_data_raw_id,
            relationship_type="FOUNDER_OF",
            source_extracted_entity_id=person_entity.id,
            source_entity_type="PERSON",
            target_extracted_entity_id=entity.id,
            target_entity_type="ORGANIZATION",
            relationship_payload_json={"title": f.get("title", "Founder")},
            created_at=datetime.utcnow()
        )
        db.add(rel)

    db.flush()
