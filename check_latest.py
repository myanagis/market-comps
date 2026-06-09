from market_comps.db.session import SessionLocal
from market_comps.db.models import PipelineRun, Organization
import json

db = SessionLocal()
runs = db.query(PipelineRun).order_by(PipelineRun.id.desc()).limit(2).all()
for r in runs:
    print(f'Run ID: {r.id}, status: {r.run_status}, pipe_id: {r.pipeline_id}, created: {r.records_created}, processed: {r.records_processed}')

orgs = db.query(Organization).order_by(Organization.id.desc()).limit(15).all()
for o in orgs:
    print(f'Org ID: {o.id}, Name: {o.name}, Type: {o.organization_type}')
