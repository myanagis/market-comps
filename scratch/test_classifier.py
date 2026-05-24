import os
import sys
import logging

logging.basicConfig(level=logging.DEBUG)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from market_comps.ingestion.classifier import classify_document
from market_comps.llm_client import LLMClient

sample_text = """
This is a startup pitch deck for Acme Corp.
We are a B2B SaaS company that helps restaurants manage inventory.
We are raising $5M in a Seed round.
Our team consists of former Google and Facebook engineers.
"""

try:
    print("Initializing LLMClient...")
    client = LLMClient()
    print("Classifying...")
    res, usage = classify_document(sample_text, client)
    print("Result:")
    print(res)
    print("Usage:")
    print(usage.model_dump())
except Exception as e:
    print(f"Error: {e}")
