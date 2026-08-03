import os
import sys

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from market_comps.ingestion.uploader_agent import UploaderChatAgent
from market_comps.db.session import SessionLocal
from market_comps.crm.company_manager import find_existing_company

agent = UploaderChatAgent()

print("Testing Uploader Agent...")
prompt = "Please add Acme Corp and Globex (globex.com) to the CRM."
print(f"User: {prompt}")

action, reply = agent.process_message(prompt, pending_companies=[], chat_history=[])
print(f"Action: {action['action']}")
print(f"Reply: {reply}")
if "extracted_companies" in action:
    print(f"Extracted: {action['extracted_companies']}")
