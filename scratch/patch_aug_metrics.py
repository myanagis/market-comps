import os
import re

file_path = r"c:\Users\micha\.gemini\antigravity\scratch\market-comps\market_comps\ingestion\company_augmentation.py"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Add extract_metrics function
extract_metrics_code = """
def extract_metrics(documents: List[Dict], target_company_name: str = "") -> Tuple[List[Dict], LLMUsage]:
    if not documents:
        from market_comps.models import LLMUsage
        return [], LLMUsage()
    client = LLMClient()
    
    doc_text_block = ""
    for i, doc in enumerate(documents):
        doc_text_block += f"\\n\\n--- Document {i} (URL: {doc['url']}) ---\\n{doc['text']}"
        
    target_directive = f"You are extracting data specifically for the company named '{target_company_name}'." if target_company_name else "You are extracting data for a target company."
        
    prompt = f\"\"\"
    {target_directive}
    Extract any specific metrics, KPIs, or financials mentioned in the text that belong to {target_company_name if target_company_name else 'the target company'}.
    Look for: Revenue, ARR (Annual Recurring Revenue), Post-money valuation, Employee count, Gross margin, or Customer count.
    
    Return a JSON object with a 'metrics' array containing objects with:
    - metric_code: One of ["revenue", "arr", "post_money_valuation", "employee_count", "gross_margin", "customer_count"]
    - value_text: The literal text value (e.g. "$10M", "1,500", "50%")
    - value_numeric: The parsed number (e.g. 10000000, 1500, 0.5) if possible
    - currency_code: e.g. "USD", if applicable
    - reporting_basis: One of ["fiscal_year", "calendar_year", "quarter", "month", "trailing_twelve_months", "run_rate", "point_in_time", "projected"]
    - observation_status: One of ["actual", "company_estimate", "company_guidance", "external_estimate"]
    - period_year: The year this applies to (e.g. 2024), if mentioned
    - source_doc_index: the integer index of the document (0-based) this was found in
    - source_excerpt: The specific quote from the text that proves this metric.
    
    Only extract clear metrics.
    
    DOCUMENTS:
    {doc_text_block}
    \"\"\"
    
    schema = {
        "type": "object",
        "properties": {
            "metrics": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "metric_code": {"type": "string"},
                        "value_text": {"type": "string"},
                        "value_numeric": {"type": ["number", "null"]},
                        "currency_code": {"type": ["string", "null"]},
                        "reporting_basis": {"type": "string"},
                        "observation_status": {"type": "string"},
                        "period_year": {"type": ["integer", "null"]},
                        "source_doc_index": {"type": "integer"},
                        "source_excerpt": {"type": "string"}
                    },
                    "required": ["metric_code", "value_text", "observation_status", "source_doc_index"]
                }
            }
        }
    }
    try:
        result, usage = client.structured_output(prompt=prompt, json_schema=schema, model=settings.default_model)
        
        metrics_list = []
        if isinstance(result, dict) and "metrics" in result:
            metrics_list = result["metrics"]
        elif isinstance(result, list):
            metrics_list = result
                    
        return metrics_list, usage
    except Exception as e:
        logger.error(f"Metric extraction failed: {e}")
        from market_comps.models import LLMUsage
        return [], LLMUsage()

def run_augmentation_pipeline(org_id: int):"""

content = content.replace("def run_augmentation_pipeline(org_id: int):", extract_metrics_code)

# 2. Add llm_model_used and source_tier to SourceDocument in run_augmentation_pipeline
content = content.replace(
    "source_url=d[\"url\"],",
    "source_url=d[\"url\"],\n                source_tier=3,\n                llm_model_used=settings.default_model,"
)

content = content.replace(
    "source_url=url,",
    "source_url=url,\n            source_tier=1,\n            llm_model_used=settings.default_model,"
)

# 3. Add Metric Processing at the end of run_augmentation_pipeline
metric_processing_code = """
        # 8. Extract and Upsert Metrics
        from market_comps.db.models import MetricType, MetricObservation, ObservationSource
        metrics_data, u_met = extract_metrics(docs_data, target_company_name=org.name)
        add_usage(u_met)
        
        # Load all metric types
        all_metric_types = db.query(MetricType).all()
        metric_code_map = {m.code: m.id for m in all_metric_types}
        
        for met in metrics_data:
            mcode = met.get("metric_code")
            if not mcode or mcode not in metric_code_map:
                continue
                
            doc_idx = met.get("source_doc_index")
            source_doc_id = None
            if doc_idx is not None and 0 <= doc_idx < len(docs_data):
                source_doc_id = docs_data[doc_idx].get("db_id")
                
            # Create Observation
            period_start, period_end = None, None
            if met.get("period_year"):
                try:
                    yr = int(met["period_year"])
                    period_start = datetime(yr, 1, 1)
                    period_end = datetime(yr, 12, 31)
                except:
                    pass
                    
            obs = MetricObservation(
                company_id=org.id,
                metric_type_id=metric_code_map[mcode],
                value_text=met.get("value_text"),
                value_numeric=met.get("value_numeric"),
                currency_code=met.get("currency_code"),
                observation_status=met.get("observation_status") or "actual",
                reporting_basis=met.get("reporting_basis"),
                period_start=period_start,
                period_end=period_end
            )
            db.add(obs)
            db.flush()
            
            if source_doc_id:
                osrc = ObservationSource(
                    metric_observation_id=obs.id,
                    source_document_id=source_doc_id,
                    source_excerpt=met.get("source_excerpt"),
                    confidence_score=0.8
                )
                db.add(osrc)
        
        db.commit()
"""

# Find the spot right before run.run_status = "SUCCESS"
content = content.replace(
    "        db.commit()\n        run.run_status = \"SUCCESS\"",
    metric_processing_code + "        run.run_status = \"SUCCESS\""
)

# Do the same for run_manual_url_augmentation
metric_processing_manual_code = """
        # 4. Extract and Upsert Metrics
        from market_comps.db.models import MetricType, MetricObservation, ObservationSource
        metrics_data, u_met = extract_metrics(docs_data, target_company_name=org.name)
        add_usage(u_met)
        
        # Load all metric types
        all_metric_types = db.query(MetricType).all()
        metric_code_map = {m.code: m.id for m in all_metric_types}
        
        for met in metrics_data:
            mcode = met.get("metric_code")
            if not mcode or mcode not in metric_code_map:
                continue
                
            # Create Observation
            period_start, period_end = None, None
            if met.get("period_year"):
                try:
                    yr = int(met["period_year"])
                    period_start = datetime(yr, 1, 1)
                    period_end = datetime(yr, 12, 31)
                except:
                    pass
                    
            obs = MetricObservation(
                company_id=org.id,
                metric_type_id=metric_code_map[mcode],
                value_text=met.get("value_text"),
                value_numeric=met.get("value_numeric"),
                currency_code=met.get("currency_code"),
                observation_status=met.get("observation_status") or "actual",
                reporting_basis=met.get("reporting_basis"),
                period_start=period_start,
                period_end=period_end
            )
            db.add(obs)
            db.flush()
            
            osrc = ObservationSource(
                metric_observation_id=obs.id,
                source_document_id=src_doc.id,
                source_excerpt=met.get("source_excerpt"),
                confidence_score=0.8
            )
            db.add(osrc)
            
        db.commit()
"""

content = content.replace(
    "        db.commit()\n        return True",
    metric_processing_manual_code + "        return True"
)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Augmentation pipeline updated with metrics extraction!")
