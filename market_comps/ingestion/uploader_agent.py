import json
import logging
from typing import Optional, Dict, Any, Tuple, List
from pydantic import BaseModel, Field

from market_comps.llm_client import LLMClient

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an intelligent CRM assistant specialized in extracting company names and details from raw text, and processing web links (news, PRs, articles).
The user will provide you with a block of text containing one or more companies to add to their CRM, OR a web link to process.
Your job is to parse the text into structured JSON actions to manage the upload process.

You have access to a list of "Pending Companies" that are currently staged to be created.
The user might provide more details about the pending companies, or they might ask you to proceed.

Valid action types:
1. `extract`: Use this when the user provides new text that contains companies. Extract the company names, domains (if any), and a brief description.
2. `process_link`: Use this when the user provides a web URL (e.g. news article, press release) AND specifies where it should be filed (e.g., Company, Investor, Market Map).
3. `update_market_map`: Use this when the user asks to add companies to a specific market map segment or public comps group.
4. `add_transaction`: Use this when the user asks to record an M&A transaction or acquisition.
5. `clarify`: Use this to ask the user a question. For example, if a company is missing a domain, ask for it. OR if the user provides a web link but doesn't specify what entity to file it to, ask them (e.g., "Where should I file this? A Company, Investor, or Market Map?").
        - When adding a company, look up its website domain and normalize its name if possible.
        - When updating a market map, you must classify the segment_type. For competitors or operating companies in a market, use 'competitors'. For public comps, use 'public_comps'. For investors, use 'investors'.
        - Guardrail: Never classify private companies as public comps. If a company is explicitly described as private, it should not be placed into a public comps segment.
6. `proceed`: Use this when the user says "yes" or "proceed" to create the pending companies.

If the user provides companies and a domain is missing, you can attempt to guess it if it is a well-known public company, otherwise just return null for the domain. Do NOT output a clarify action just because the domain is missing. The backend will attempt to find the domain automatically via search.
If the user indicates a company is public, you should extract its ticker_symbol, stock_exchange, and set ownership_type to "PUBLIC".
If the text describes an investment firm, VC, PE firm, or similar, set organization_type to "INVESTOR". Otherwise, set it to "COMPANY".

When validation rules are provided, you MUST adhere to them. If a field listed in `required_fields` is missing or cannot be inferred from the text, you MUST output a `clarify` action to ask the user for it, and DO NOT output an `extract` action.
If `extract_parameters` is provided, you should attempt to extract those specific fields into the `parameters` dictionary.

IMPORTANT: You MUST ONLY reply with a JSON object format matching the required schema. Do not include markdown formatting or extra text.
"""

ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["extract", "clarify", "proceed", "process_link", "update_market_map", "add_transaction"],
            "description": "The type of action to perform based on user input."
        },
        "message": {
            "type": "string",
            "description": "Conversational reply or question to the user. Explain what you are doing or what you need."
        },
        "extracted_companies": {
            "type": "array",
            "description": "List of companies extracted from the text (used when action is 'extract').",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "domain": {"type": ["string", "null"]},
                    "description": {"type": ["string", "null"]},
                    "ticker_symbol": {"type": ["string", "null"]},
                    "stock_exchange": {"type": ["string", "null"]},
                    "ownership_type": {"type": ["string", "null"]},
                    "organization_type": {"type": ["string", "null"], "enum": ["COMPANY", "INVESTOR", None]},
                    "parameters": {
                        "type": "object",
                        "description": "Any additional dynamic parameters requested to be extracted (e.g., founders, founded_year, check_size).",
                        "additionalProperties": True
                    },
                    "canonical_source_name": {
                        "type": ["string", "null"],
                        "description": "If the user mentions a specific source they found this from (e.g. 'CT Business Registry', 'Luma Demo Day'), extract it here. Use the closest matching valid canonical source if you recognize it."
                    }
                },
                "required": ["name"]
            }
        },
        "url": {
            "type": ["string", "null"],
            "description": "The web URL to process (used when action is 'process_link' or 'process_event_link')."
        },
        "target_entity_type": {
            "type": ["string", "null"],
            "enum": ["COMPANY", "INVESTOR", "MARKET_MAP", None],
            "description": "The type of entity to file the web link data to."
        },
        "target_entity_name": {
            "type": ["string", "null"],
            "description": "The specific name of the entity to file the web link data to."
        },
        "market_map_update": {
            "type": ["object", "null"],
            "description": "Used when action is 'update_market_map'.",
            "properties": {
                "market_name": {"type": "string", "description": "The name of the market map or comparison set."},
                "segment_name": {"type": "string", "description": "The name of the segment within the market map, if specified."},
                "segment_type": {
                    "type": "string",
                    "enum": ["competitors", "public_comps", "investors", "other"],
                    "description": "The type of the segment (e.g. competitors, public_comps)."
                },
                "companies": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "domain": {"type": ["string", "null"]},
                            "ticker_symbol": {"type": ["string", "null"]},
                            "stock_exchange": {"type": ["string", "null"]},
                            "ownership_type": {"type": ["string", "null"]}
                        },
                        "required": ["name"]
                    },
                    "description": "List of companies to add to the market map."
                },
                "notes": {"type": ["string", "null"], "description": "Any additional comments or differentiation notes provided by the user."}
            },
            "required": ["market_name", "segment_type", "companies"]
        },
        "transaction_details": {
            "type": ["object", "null"],
            "description": "Used when action is 'add_transaction'.",
            "properties": {
                "acquirer": {"type": "string"},
                "target": {"type": "string"},
                "price": {"type": ["number", "null"], "description": "The price of the transaction, if specified (e.g. 500000000 for 500M)."},
                "currency": {"type": ["string", "null"], "description": "Currency code like USD."},
                "notes": {"type": ["string", "null"]}
            },
            "required": ["acquirer", "target"]
        }
    },
    "required": ["action", "message"]
}

class UploaderChatAgent:
    def __init__(self, model: str = "openai/gpt-4o"):
        self.client = LLMClient(model=model)
        
    def process_message(
        self, 
        user_message: str, 
        pending_companies: List[Dict[str, Any]], 
        chat_history: Optional[list] = None,
        validation_rules: Optional[Dict[str, Any]] = None,
        existing_sources: Optional[List[str]] = None
    ) -> Tuple[Dict[str, Any], str]:
        """
        Processes a user message along with the current pending companies.
        Returns a tuple: (action_dict, llm_reply_message)
        """
        # Format chat history
        history_str = "No recent chatting history."
        if chat_history:
            history_lines = []
            for msg in chat_history[-5:]:
                role = msg.get("role", "unknown").upper()
                content = msg.get("content", "")
                history_lines.append(f"{role}: {content}")
            history_str = "\n".join(history_lines)
            
        pending_str = json.dumps(pending_companies, indent=2) if pending_companies else "[]"
        
        rules_str = ""
        if validation_rules:
            rules_str = f"VALIDATION RULES:\n{json.dumps(validation_rules, indent=2)}\n"
            
        sources_str = ""
        if existing_sources:
            sources_str = f"EXISTING CANONICAL SOURCES (Use these for fuzzy matching):\n{json.dumps(existing_sources, indent=2)}\n"
        
        prompt = f"""\
RECENT CHAT HISTORY:
{history_str}

CURRENT PENDING COMPANIES STAGED FOR CREATION:
{pending_str}

{rules_str}
{sources_str}
USER REQUEST:
"{user_message}"

Based on the rules, what is the appropriate JSON action?
"""
        try:
            parsed_json, _ = self.client.structured_output(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                json_schema=ACTION_SCHEMA,
                temperature=0.1
            )
            
            # Defensive unpacking
            if "action" not in parsed_json:
                for possible_action in ["extract", "clarify", "proceed", "process_link", "process_event_link"]:
                    if possible_action in parsed_json:
                        inner_data = parsed_json[possible_action]
                        if isinstance(inner_data, dict):
                            parsed_json = {
                                "action": possible_action,
                                "message": inner_data.get("message", "Processing..."),
                                "extracted_companies": inner_data.get("extracted_companies", []),
                                "url": inner_data.get("url"),
                                "target_entity_type": inner_data.get("target_entity_type"),
                                "target_entity_name": inner_data.get("target_entity_name")
                            }
                            # Ensure parameters is passed through if it exists in extracted_companies
                            if possible_action == "extract":
                                for comp in parsed_json["extracted_companies"]:
                                    if "parameters" not in comp:
                                        comp["parameters"] = {}
                        break
                        
            # Ensure parameters exists in root parsing
            if parsed_json.get("action") == "extract":
                for comp in parsed_json.get("extracted_companies", []):
                    if "parameters" not in comp:
                        comp["parameters"] = {}
                        
            message = parsed_json.get("message", "Processing request...")
            return parsed_json, message
            
        except Exception as e:
            logger.error(f"Error parsing uploader agent output: {e}")
            return {"action": "clarify", "message": f"I had trouble parsing that. Can you rephrase? (Error: {e})"}, "I had trouble parsing that."
