import json
import logging
from typing import Optional, Dict, Any, Tuple, List
from pydantic import BaseModel, Field

from market_comps.llm_client import LLMClient

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an intelligent CRM assistant specialized in extracting company names and details from raw text.
The user will provide you with a block of text containing one or more companies to add to their CRM.
Your job is to parse the text into structured JSON actions to manage the upload process.

You have access to a list of "Pending Companies" that are currently staged to be created.
The user might provide more details about the pending companies, or they might ask you to proceed.

Valid action types:
1. `extract`: Use this when the user provides new text that contains companies. Extract the company names, domains (if any), and a brief description.
2. `clarify`: Use this to ask the user a question. For example, if a company is missing a domain, ask the user for it.
3. `proceed`: Use this when the user says "yes" or "proceed" to create the pending companies.

If the user provides companies, but one or more are missing a domain/website, you SHOULD use the `clarify` action to ask for the domain, because the AI augmentation pipeline works best with a website.
However, if the user explicitly says they don't know the domain, or tells you to proceed anyway, use the `proceed` action.

IMPORTANT: You MUST ONLY reply with a JSON object format matching the required schema. Do not include markdown formatting or extra text.
"""

ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["extract", "clarify", "proceed"],
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
                    "description": {"type": ["string", "null"]}
                },
                "required": ["name"]
            }
        }
    },
    "required": ["action", "message"]
}

class UploaderChatAgent:
    def __init__(self, model: str = "openai/gpt-4o"):
        self.client = LLMClient(model=model)
        
    def process_message(self, user_message: str, pending_companies: List[Dict[str, Any]], chat_history: Optional[list] = None) -> Tuple[Dict[str, Any], str]:
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
        
        prompt = f"""\
RECENT CHAT HISTORY:
{history_str}

CURRENT PENDING COMPANIES STAGED FOR CREATION:
{pending_str}

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
                for possible_action in ["extract", "clarify", "proceed"]:
                    if possible_action in parsed_json:
                        inner_data = parsed_json[possible_action]
                        if isinstance(inner_data, dict):
                            parsed_json = {
                                "action": possible_action,
                                "message": inner_data.get("message", "Processing..."),
                                "extracted_companies": inner_data.get("extracted_companies", [])
                            }
                        break
                        
            message = parsed_json.get("message", "Processing request...")
            return parsed_json, message
            
        except Exception as e:
            logger.error(f"Error parsing uploader agent output: {e}")
            return {"action": "clarify", "message": f"I had trouble parsing that. Can you rephrase? (Error: {e})"}, "I had trouble parsing that."
