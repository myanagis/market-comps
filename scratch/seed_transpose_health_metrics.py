import os
import sys
from datetime import datetime

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from market_comps.db.session import SessionLocal
from market_comps.db.models import (
    Organization, SourceDocument, DocumentText, MetricType, 
    MetricObservation, ObservationSource
)

def seed_transpose_health():
    db = SessionLocal()
    try:
        org = db.query(Organization).filter(Organization.name.ilike("%Transpose Health%")).first()
        if not org:
            print("Transpose Health not found.")
            return

        rev_metric = db.query(MetricType).filter_by(code="revenue").first()
        if not rev_metric:
            print("Revenue metric type not found.")
            return

        # Create Fake Source 1: TechCrunch Article
        src1 = SourceDocument(
            pipeline_run_id=1, # Fake run ID, usually created by ingestion but for a mock we just need any valid int or nullable. Oh wait, pipeline_run_id is nullable=False!
            source_type="MANUAL_UPLOAD",
            document_type="WEB_PAGE",
            title="TechCrunch: Transpose Health crosses $15M in 2024",
            source_url="https://techcrunch.com/fake-transpose-health-revenue",
            source_tier=3,
            llm_model_used="openai/gpt-4o",
            created_at=datetime(2024, 12, 1)
        )
        # Let's bypass pipeline_run_id by just finding the first pipeline_run or creating one.
        from market_comps.db.models import PipelineRun
        run = db.query(PipelineRun).first()
        if not run:
            run = PipelineRun(run_status="SUCCESS")
            db.add(run)
            db.flush()
        src1.pipeline_run_id = run.id
        
        db.add(src1)
        db.flush()
        
        db.add(DocumentText(source_document_id=src1.id, data_type="PAGE_TEXT", raw_content="Transpose Health recently announced their 2024 revenue hit $15M."))
        
        obs1 = MetricObservation(
            company_id=org.id,
            metric_type_id=rev_metric.id,
            value_text="$15M",
            value_numeric=15000000.0,
            currency_code="USD",
            observation_status="actual",
            reporting_basis="fiscal_year",
            period_start=datetime(2024, 1, 1),
            period_end=datetime(2024, 12, 31)
        )
        db.add(obs1)
        db.flush()
        
        db.add(ObservationSource(observation_id=obs1.id, source_id=src1.id, source_excerpt="2024 revenue hit $15M", relationship_type="primary"))

        # Create Fake Source 2: Conflicting Internal Leak
        src2 = SourceDocument(
            pipeline_run_id=run.id,
            source_type="MANUAL_UPLOAD",
            document_type="WEB_PAGE",
            title="Business Insider: Internal docs show Transpose Health at $12M",
            source_url="https://businessinsider.com/fake-transpose-leak",
            source_tier=4,
            llm_model_used="openai/gpt-4o",
            created_at=datetime(2024, 12, 5)
        )
        db.add(src2)
        db.flush()
        
        db.add(DocumentText(source_document_id=src2.id, data_type="PAGE_TEXT", raw_content="Internal documents leaked to BI suggest Transpose actually made $12M in 2024, contrary to public claims."))
        
        obs2 = MetricObservation(
            company_id=org.id,
            metric_type_id=rev_metric.id,
            value_text="$12M",
            value_numeric=12000000.0,
            currency_code="USD",
            observation_status="external_estimate",
            reporting_basis="fiscal_year",
            period_start=datetime(2024, 1, 1),
            period_end=datetime(2024, 12, 31)
        )
        db.add(obs2)
        db.flush()
        
        db.add(ObservationSource(observation_id=obs2.id, source_id=src2.id, source_excerpt="actually made $12M in 2024", relationship_type="primary"))

        # Create Fake Source 3: Projected 2025 Revenue
        src3 = SourceDocument(
            pipeline_run_id=run.id,
            source_type="MANUAL_UPLOAD",
            document_type="PDF",
            title="Transpose Health Investor Deck 2024 Q4",
            source_url="https://docsend.com/transpose-deck-q4",
            source_tier=1, # Highly trusted
            llm_model_used="anthropic/claude-3-5-sonnet",
            created_at=datetime(2025, 1, 10)
        )
        db.add(src3)
        db.flush()
        
        db.add(DocumentText(source_document_id=src3.id, data_type="PAGE_TEXT", raw_content="Looking forward to 2025, we project our revenue to grow aggressively to $35M based on our current pipeline."))
        
        obs3 = MetricObservation(
            company_id=org.id,
            metric_type_id=rev_metric.id,
            value_text="$35M",
            value_numeric=35000000.0,
            currency_code="USD",
            observation_status="company_estimate",
            reporting_basis="projected",
            period_start=datetime(2025, 1, 1),
            period_end=datetime(2025, 12, 31)
        )
        db.add(obs3)
        db.flush()
        
        db.add(ObservationSource(observation_id=obs3.id, source_id=src3.id, source_excerpt="project our revenue to grow aggressively to $35M", relationship_type="primary"))

        db.commit()
        print("Successfully seeded Transpose Health with fake metrics and sources!")

    except Exception as e:
        db.rollback()
        print(f"Error: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    seed_transpose_health()
