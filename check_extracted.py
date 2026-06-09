from market_comps.db.session import SessionLocal
from market_comps.db.models import ExtractedEntity
import json

db = SessionLocal()
entities = db.query(ExtractedEntity).order_by(ExtractedEntity.id.desc()).limit(5).all()

for e in entities:
    print(f"ID: {e.id}, Type: {e.entity_type}, Name: {e.raw_name}")
    print(json.dumps(e.extracted_payload_json, indent=2))
    print("-" * 50)
