import json
import logging
from typing import Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field

from market_comps.llm_client import LLMClient
from market_comps.waterfall.models import CapTable, ExitScenario, SecurityClass

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an expert venture capital analyst and cap table manager.
The user will describe a sequence of funding events, changes to their cap table, or an exit scenario.
Your job is to parse their natural language description into structured JSON actions to update the model.

You have access to the current state of the CapTable and the Recent Chat History. Check if the user is ADDING, EDITING, or REMOVING a security, or UPDATING the exit scenario.

If the user provides an event but vital information is missing, respond with a conversational question asking for the missing info. Valid triggers for asking questions:
- Missing amount raised or valuation cap for a SAFE.
- Missing share price or valuation for a priced round.
- If a Preferred equity security is participating, you MUST ask if there is a participation cap (if not provided). NOTE: Convertible Notes and SAFEs do NOT have participation (do NOT set `is_participating` for them), but you CAN ask if they have a liquidation preference.
- If a security is a convertible note/debt with an interest rate, you MUST ask if the interest is simple or compounding (if not provided).

If you have enough information, emit a structured JSON object with the appropriate action type.

Valid action types:
1. `add_security`
2. `edit_security` 
3. `remove_security`
4. `update_exit`
5. `ask_question` (Use this if you need more info, and provide your question in the `message` field).

For `add_security` or `edit_security`, you must provide the fields to populate `SecurityClass`. You MUST include `series_name` and `security_type`.
For `remove_security`, provide the `series_name`.
For `update_exit`, provide `exit_value_usd` and optionally `exit_date`.

IMPORTANT: You MUST ONLY reply with a JSON object format matching the required schema. Do not include markdown formatting or extra text.

Example format:
{
  "action": "add_security",
  "message": "Adding the $5M SAFE now.",
  "security": {
    "series_name": "Seed SAFE",
    "security_type": "safe",
    ...
  }
}
"""

# JSON schema we want the model to output
ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["add_security", "edit_security", "remove_security", "update_exit", "ask_question"],
            "description": "The type of action to perform based on user input."
        },
        "message": {
            "type": "string",
            "description": "Conversational reply or question to the user (used extensively for ask_question)."
        },
        "security": {
            "type": "object",
            "description": "Fields matching the SecurityClass model. Use for add/edit.",
            "properties": {
                "series_name": {"type": "string"},
                "security_type": {"type": "string", "enum": ["common", "preferred", "safe", "convertible_note", "warrant", "option"]},
                "close_date": {"type": "string"},
                "maturity_date": {"type": "string"},
                "seniority": {"type": "integer"},
                "issue_price": {"type": "number"},
                "total_investment_usd": {"type": "number"},
                "total_shares": {"type": "integer"},
                "liquidation_preference_multiple": {"type": "number"},
                "is_participating": {"type": "boolean"},
                "participation_cap_multiple": {"type": "number"},
                "discount_rate": {"type": "number"},
                "valuation_cap": {"type": "number"},
                "interest_rate": {"type": "number"},
                "is_interest_compounding": {"type": "boolean"}
            },
            "required": ["series_name", "security_type"]
        },
        "series_to_remove": {
            "type": "string",
            "description": "Name of the series to remove if action is remove_security."
        },
        "exit_scenario": {
            "type": "object",
            "properties": {
                "exit_value_usd": {"type": "number"},
                "exit_date": {"type": "string"}
            }
        }
    },
    "required": ["action", "message"]
}


class WaterfallChatAgent:
    def __init__(self, model: str = "openai/gpt-4o"):
        self.client = LLMClient(model=model)
        
    def process_message(self, user_message: str, current_cap_table: CapTable, chat_history: Optional[list] = None) -> Tuple[Dict[str, Any], str]:
        """
        Processes a user message along with the current state of the cap table.
        Returns a tuple: (action_dict, llm_reply_message)
        """
        # Create a state summary to send to the LLM
        cap_table_json = current_cap_table.model_dump_json(indent=2)
        
        # Format chat history
        history_str = "No recent chatting history."
        if chat_history:
            history_lines = []
            for msg in chat_history[-5:]: # Keep last 5 messages for context
                role = msg.get("role", "unknown").upper()
                content = msg.get("content", "")
                history_lines.append(f"{role}: {content}")
            history_str = "\n".join(history_lines)
        
        prompt = f"""\
RECENT CHAT HISTORY:
{history_str}

CURRENT CAP TABLE STATE:
{cap_table_json}

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
            
            # Defensive unpacking in case LLM wraps the action (e.g. {"add_security": {...}})
            if "action" not in parsed_json:
                for possible_action in ["add_security", "edit_security", "remove_security", "update_exit", "ask_question"]:
                    if possible_action in parsed_json:
                        inner_data = parsed_json[possible_action]
                        if isinstance(inner_data, dict):
                            parsed_json = {
                                "action": possible_action,
                                "message": inner_data.get("message", "Processing..."),
                                "security": inner_data
                            }
                        break
                        
            message = parsed_json.get("message", "I have updated the model based on your request.")
            return parsed_json, message
            
        except Exception as e:
            logger.error(f"Error parsing chat agent output: {e}")
            return {"action": "ask_question", "message": f"I had trouble parsing that. Can you rephrase? (Error: {e})"}, "I had trouble parsing that."
