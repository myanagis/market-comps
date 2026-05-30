from sqlalchemy.orm import Session
from market_comps.db.session import SessionLocal
from market_comps.db.models import Pipeline, PipelineRun
from market_comps.ingestion.pipeline_runner import run_pipeline

def test_sec_form_d():
    db: Session = SessionLocal()
    
    # 1. Create a mock SEC pipeline config
    pipeline = Pipeline(
        pipeline_name="SEC Form D Scraper",
        connector_type="SEC_DAILY_INDEX",
        parser_type="SEC_FORM_D_XML",
        normalizer_type="SEC_FORM_D_TO_ORGS",
        config_json={
            "days_back": 1,
            "max_filings_to_process": 3,
            "updater_type": "SEC_FORM_D_UPDATER"
        }
    )
    db.add(pipeline)
    db.commit()
    db.refresh(pipeline)

    print(f"Created Pipeline {pipeline.id}")

    try:
        run = run_pipeline(db, pipeline.id)
        print(f"Pipeline Run Completed! Status: {run.run_status}")
        print(f"Stats: {run.logs_json}")
    finally:
        # Cleanup (optional if we want to keep them)
        pass

if __name__ == "__main__":
    test_sec_form_d()
