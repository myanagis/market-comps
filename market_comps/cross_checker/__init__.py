# market_comps/cross_checker/__init__.py
from market_comps.cross_checker.cross_checker import LLMChorus, ChorusResult, ModelResponse

# Legacy aliases so any existing code keeps working
CrossChecker = LLMChorus
CrossCheckResult = ChorusResult

__all__ = ["LLMChorus", "ChorusResult", "ModelResponse", "CrossChecker", "CrossCheckResult"]
