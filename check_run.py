from market_comps.db.session import SessionLocal
from market_comps.db.models import PipelineRun, PipelineRunStep, SourceDocument, DocumentText
import json

db = SessionLocal()
run = db.query(PipelineRun).order_by(PipelineRun.id.desc()).first()
print(f'Run ID: {run.id}, Status: {run.run_status}')
for step in run.steps:
    print(f'Step {step.step_name}: status={step.status}, output_count={step.output_count}')

# SEC pipeline saves extracted data in logs_json maybe? Or returns it.
# Actually, the pipeline saves things to `PipelineRunStep` or we can just print the recent extracted objects from SEC pipeline.
# Let's see what is in run.logs_json
if run.logs_json:
    print(json.dumps(run.logs_json, indent=2)[:500])

print("DONE")
