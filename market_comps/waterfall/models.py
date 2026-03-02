from enum import Enum
from pydantic import BaseModel, Field, model_validator
from typing import Optional, List, Dict
from datetime import date

class SecurityType(str, Enum):
    COMMON = "common"
    CONVERTIBLE_NOTE = "convertible note"
    SAFE = "safe"
    PREFERRED = "preferred"

class Security(BaseModel):
    security_type: SecurityType
    name: str = Field(description="Name or label for the security (e.g., 'Series A', 'Seed SAFE')")
    date: date
    total_shares: Optional[int] = None
    fully_diluted_shares: Optional[int] = None
    capital_raised: float = 0.0
    liquidity_preference: float = Field(default=1.0, description="Liquidity preference multiple (e.g., 1.0x)")

    # Convertible Note & SAFE specifics
    interest_rate: Optional[float] = None
    is_compounding: Optional[bool] = None
    maturity_date: Optional[date] = None
    pre_money_cap: Optional[float] = None
    discount: Optional[float] = None

    # Priced Rounds (Preferred / Equity)
    pre_money: Optional[float] = None
    post_money: Optional[float] = None
    post_money_options_pool_percentage: Optional[float] = None

    @model_validator(mode='after')
    def validate_security_fields(self):
        if self.security_type == SecurityType.CONVERTIBLE_NOTE:
            if self.pre_money_cap is None:
                raise ValueError("Pre-money cap is required for convertible notes")
        
        if self.security_type == SecurityType.PREFERRED:
            if self.pre_money is None or self.post_money is None:
                raise ValueError("Pre-money and post-money are required for priced rounds")
            
        return self

class ExitScenario(BaseModel):
    exit_value: float
    exit_date: date

class ExitResult(BaseModel):
    payouts: Dict[str, float]
    moic: Dict[str, float]
    explanations: List[str]

class WaterfallModel(BaseModel):
    securities: List[Security]
    
    def calculate_exit_waterfall(self, exit_scenario: ExitScenario) -> ExitResult:
        """
        STUB: Calculate the distribution of proceeds to each security in an exit scenario.
        
        This will need to handle:
        1. Determining chronological order or seniority
        2. Converting notes / SAFEs
        3. Paying out liquidity preferences
        4. Distributing remaining proceeds to common and participating preferred
        """
        # TODO: Implement exit waterfall logic guided by user
        from market_comps.waterfall.calculator import WaterfallCalculator
        return WaterfallCalculator.calculate_exit_waterfall(self, exit_scenario)

