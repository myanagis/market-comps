import hashlib
import logging
from datetime import datetime
from urllib.parse import urljoin

from sqlalchemy.orm import Session

from market_comps.db.models import Pipeline, PipelineRun, SourceDocument, DocumentText, ExtractionJob
from market_comps.ingestion.interfaces import BaseFetcher, BasePreparer, BaseExtractor, BaseNormalizer, BaseUpdater
from market_comps.ingestion.scraper import fetch_page_text
from market_comps.ingestion.extractor import extract_entities_from_text, extract_profile_detail
from market_comps.ingestion.reconciler import reconcile_all

logger = logging.getLogger(__name__)

class GenericHTTPFetcher(BaseFetcher):
    def fetch_data(self, db: Session, run: PipelineRun, pipeline: Pipeline) -> str:
        url = pipeline.source_url or ""
        logger.info(f"[GenericHTTPFetcher] Fetching {url}")
        text_content = fetch_page_text(url)
        return text_content

class GenericHTMLPreparer(BasePreparer):
    def prepare_raw_data(self, db: Session, run: PipelineRun, pipeline: Pipeline, raw_data: str) -> DocumentText:
        content_hash = hashlib.sha256(raw_data.encode()).hexdigest()
        
        source_doc = SourceDocument(
            pipeline_run_id=run.id,
            document_type="WEB_PAGE",
            source_url=pipeline.source_url,
            content_hash=content_hash,
        )
        db.add(source_doc)
        db.flush()

        doc_text = DocumentText(
            source_document_id=source_doc.id,
            data_type="PAGE_TEXT",
            raw_content=raw_data,
            content_hash=content_hash,
        )
        db.add(doc_text)
        db.flush()
        return doc_text

class GenericLLMExtractor(BaseExtractor):
    def extract_attributes(self, db: Session, run: PipelineRun, pipeline: Pipeline, prepared_data: DocumentText) -> dict:
        config = pipeline.config_json or {}
        is_deep_scrape = config.get("deep_scrape", False)
        
        job = ExtractionJob(
            pipeline_run_id=run.id,
            document_text_id=prepared_data.id,
            schema_name="PROGRAM_COMPANY_SCHEMA",
            status="IN_PROGRESS",
            started_at=datetime.utcnow()
        )
        db.add(job)
        db.flush()

        extraction_result = extract_entities_from_text(
            db, run, job, prepared_data, pipeline.connector_type, config
        )
        
        # Deep scrape logic
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

                profile_url = urljoin(pipeline.source_url, profile_path)
                logger.info(f"[GenericLLMExtractor] Deep scrape ({i+1}/{len(companies_raw)}): {profile_url}")

                try:
                    profile_text = fetch_page_text(profile_url)
                    profile_hash = hashlib.sha256(profile_text.encode()).hexdigest()

                    profile_doc = SourceDocument(
                        pipeline_run_id=run.id,
                        document_type="WEB_PAGE",
                        source_url=profile_url,
                        content_hash=profile_hash,
                    )
                    db.add(profile_doc)
                    db.flush()

                    profile_text_obj = DocumentText(
                        source_document_id=profile_doc.id,
                        data_type="PROFILE_TEXT",
                        raw_content=profile_text,
                        content_hash=profile_hash,
                    )
                    db.add(profile_text_obj)
                    db.flush()

                    profile_job = ExtractionJob(
                        pipeline_run_id=run.id,
                        document_text_id=profile_text_obj.id,
                        schema_name="PROFILE_DETAIL_SCHEMA",
                        status="IN_PROGRESS",
                        started_at=datetime.utcnow()
                    )
                    db.add(profile_job)
                    db.flush()

                    deep_res = extract_profile_detail(db, run, profile_job, profile_text_obj, c, config)
                    
                    # Merge deep_res into c
                    if deep_res.get("company_profile"):
                        c.update(deep_res["company_profile"])
                        deep_logs.append({"company": c.get("name"), "status": "success"})
                    
                except Exception as e:
                    logger.error(f"Error deep scraping {profile_url}: {e}")
                    deep_logs.append({"company": c.get("name"), "status": f"error: {str(e)}"})

            extraction_result["deep_logs"] = deep_logs

        return extraction_result

class InvestorPortfolioNormalizer(BaseNormalizer):
    def normalize_data(self, db: Session, run: PipelineRun, pipeline: Pipeline, extracted_data: dict) -> dict:
        # For the generic web pipeline, normalizer and updater were previously combined in reconcile_all
        # So we'll just pass the extraction_result through
        return extracted_data

class GenericCanonicalUpdater(BaseUpdater):
    def update_records(self, db: Session, run: PipelineRun, pipeline: Pipeline, normalized_data: dict) -> dict:
        # reconcile_all handles both normalizing logic and DB updates for generic data
        return reconcile_all(db, run, pipeline)
