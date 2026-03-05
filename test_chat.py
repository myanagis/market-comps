from market_comps.waterfall.models import CapTable
from market_comps.waterfall.chat_agent import WaterfallChatAgent
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

ct = CapTable()
agent = WaterfallChatAgent()

print("Sending request to LLM...")
res = agent.process_message("Add a $5M Series A at $20M pre-money", ct)
print("Response:", res)
