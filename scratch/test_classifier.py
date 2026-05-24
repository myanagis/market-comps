import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from market_comps.llm_client import LLMClient
from market_comps.ingestion.classifier import classify_document, get_recommended_schemas

def main():
    print("Testing document classification...")
    llm = LLMClient()
    text = """
    Acme Corp Series A Pitch Deck. 
    We are a fast growing B2B SaaS startup revolutionizing the cloud.
    Founders: John Doe (CEO), Jane Smith (CTO)
    Revenue: $2M ARR.
    Seeking: $10M Series A at $40M pre-money valuation.
    """
    
    result, usage = classify_document(text, llm)
    print("Classification Result:")
    import json
    print(json.dumps(result, indent=2))
    print("Usage:", usage)
    
    doc_class = result.get("document_type")
    schemas = get_recommended_schemas(doc_class)
    print(f"\nRecommended schemas for {doc_class}:")
    for s in schemas:
        print(f" - {s}")

if __name__ == "__main__":
    main()
