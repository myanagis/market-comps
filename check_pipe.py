from market_comps.db.session import SessionLocal
from market_comps.db.models import Pipeline
db = SessionLocal()
p = db.query(Pipeline).filter_by(id=10).first()
if p:
    print(f'Pipeline 10: {p.pipeline_name}, connector: {p.connector_type}, normalizer: {p.normalizer_type}')
else:
    print("Pipeline 10 not found")
