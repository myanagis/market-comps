from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field
from enum import Enum
import uuid
import datetime

class SecurityType(str, Enum):
    COMMON = "common"
    PREFERRED = "preferred"
    SAFE = "safe"
    CONVERTIBLE_NOTE = "convertible_note"
    WARRANT = "warrant"
    OPTION = "option"

class SecurityClass(BaseModel):
    """
    Represents a specific class or series of security in the cap table.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    series_name: str = Field(description="Name of the round/series (e.g., 'Series A', 'Seed SAFE')")
    security_type: SecurityType = Field(description="Type of the security")
    
    # Dates
    close_date: Optional[str] = Field(None, description="Date the round closed (YYYY-MM-DD)")
    maturity_date: Optional[str] = Field(None, description="Maturity date for Notes/SAFEs (YYYY-MM-DD)")
    
    # Economics & Ranking
    seniority: int = Field(default=1, description="Payout ranking relative to others. Lower number = paid earlier (e.g. 1 is senior to 2). Defaults to 1.")
    
    # Preferred/Common properties
    issue_price: Optional[float] = Field(None, description="Price per share (if applicable)")
    total_investment_usd: Optional[float] = Field(None, description="Total dollars invested in this class")
    total_shares: Optional[int] = Field(None, description="Number of shares issued in this class")
    
    # Liquidation Pref & Participation
    liquidation_preference_multiple: float = Field(default=1.0, description="Multiple of investment returned before common (e.g., 1.0x)")
    is_participating: bool = Field(default=False, description="Does this preferred class participate with common after its liq pref?")
    participation_cap_multiple: Optional[float] = Field(None, description="The max total return multiple if participating (e.g. 2.0x). None = uncapped.")
    
    # SAFEs / Convertibles specific
    discount_rate: Optional[float] = Field(None, description="Discount rate for converting to preferred (e.g., 0.20 for 20% discount)")
    valuation_cap: Optional[float] = Field(None, description="Valuation cap for converting to preferred")
    interest_rate: Optional[float] = Field(None, description="Annual interest rate (e.g. 0.08 for 8%)")
    is_interest_compounding: bool = Field(default=False, description="Whether the interest compounds annually")

    def __str__(self):
        return f"{self.series_name} ({self.security_type.value})"


class CapTable(BaseModel):
    """
    Holds the complete state of the company's capital stack.
    """
    company_name: str = "My Startup"
    securities: List[SecurityClass] = Field(default_factory=list)
    
    # Pre-money and post-money references for context (useful for the LLM)
    latest_post_money_valuation: Optional[float] = None

    def add_security(self, security: SecurityClass):
        self.securities.append(security)

    def remove_security(self, series_name: str) -> bool:
        original_count = len(self.securities)
        self.securities = [s for s in self.securities if s.series_name.lower() != series_name.lower()]
        return len(self.securities) < original_count

    def get_security(self, series_name: str) -> Optional[SecurityClass]:
        for s in self.securities:
            if s.series_name.lower() == series_name.lower():
                return s
        return None

class ExitScenario(BaseModel):
    """
    Represents an arbitrary exit event.
    """
    exit_value_usd: float = Field(..., description="Total enterprise value at exit")
    exit_date: Optional[str] = Field(None, description="Date of exit (YYYY-MM-DD)")
    
    # To be populated by calculator
    payouts: Dict[str, float] = Field(default_factory=dict, description="Map of SecurityClass ID to total payout amount")
