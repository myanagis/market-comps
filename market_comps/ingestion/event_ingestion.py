import json
import logging
from typing import Dict, Any, List

from market_comps.llm_client import LLMClient
from market_comps.config import settings

logger = logging.getLogger(__name__)

EVENT_PARSER_SCHEMA = {
    "type": "object",
    "properties": {
        "event_name": {"type": "string"},
        "start_at": {"type": ["string", "null"], "description": "ISO 8601 date string"},
        "end_at": {"type": ["string", "null"], "description": "ISO 8601 date string"},
        "location": {"type": ["string", "null"]},
        "event_type": {"type": ["string", "null"], "enum": ["conference", "demo_day", "webinar", "meetup", "pitch_event", None]},
        "companies": {
            "type": "array",
            "description": "List of companies presenting or attending.",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "domain": {"type": ["string", "null"]},
                    "description": {"type": ["string", "null"]},
                    "role": {"type": "string", "enum": ["presenter", "attendee", "sponsor", "organizer"]}
                },
                "required": ["name", "role"]
            }
        },
        "investors": {
            "type": "array",
            "description": "List of investors attending, speaking, or organizing.",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "domain": {"type": ["string", "null"]},
                    "description": {"type": ["string", "null"]},
                    "role": {"type": "string", "enum": ["organizer", "speaker", "attendee", "sponsor"]}
                },
                "required": ["name", "role"]
            }
        }
    },
    "required": ["event_name"]
}

class EventScraperAgent:
    def __init__(self, model: str = settings.default_model):
        self.client = LLMClient(model=model)
        
    def process_event_url(self, url: str) -> Dict[str, Any]:
        """Scrape the URL and extract event details, companies, and investors."""
        # Fast local scraping via Jina
        import requests
        try:
            response = requests.get(f"https://r.jina.ai/{url}", timeout=15)
            text = response.text[:20000] if response.status_code == 200 else ""
        except Exception as e:
            logger.error(f"Failed to scrape {url}: {e}")
            text = ""
            
        if not text:
            raise ValueError(f"Could not retrieve content from {url}")
            
        prompt = f"""\
You are an expert event data parser. The following text was scraped from an event page (e.g. Luma, Eventbrite).
Please extract the event details, along with any startups/companies mentioned (e.g., as presenters, sponsors, attendees) and any investors/VCs mentioned (e.g., as organizers, speakers, attendees).

EVENT PAGE URL: {url}

SCRAPED TEXT:
{text}
"""
        
        result, usage = self.client.structured_output(
            prompt=prompt,
            json_schema=EVENT_PARSER_SCHEMA
        )
        
        return result
