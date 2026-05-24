import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from market_comps.db.session import get_db
from market_comps.db.models import SourceDocument, DocumentText, ExtractionJob, ExtractedEntity, EntityMatch

db = next(get_db())

try:
    docs = db.query(SourceDocument).join(DocumentText).join(ExtractionJob).join(ExtractedEntity).join(EntityMatch).filter(
        EntityMatch.canonical_entity_type == "Organization",
        EntityMatch.canonical_entity_id == "1"
    ).distinct().all()
    print("Success! Docs:", docs)
except Exception as e:
    import traceback
    traceback.print_exc()
