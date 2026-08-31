import yfinance as yf
import time
import logging
from typing import Dict, Any, Optional
from decimal import Decimal

logger = logging.getLogger(__name__)

class YahooFinanceClient:
    def __init__(self):
        self._last_call_time = 0.0
        self._throttle_seconds = 1.0

    def _throttle(self):
        now = time.time()
        elapsed = now - self._last_call_time
        if elapsed < self._throttle_seconds:
            time.sleep(self._throttle_seconds - elapsed)
        self._last_call_time = time.time()

    def fetch_financial_metrics(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Fetches TTM (Trailing Twelve Months) or latest available metrics for a ticker.
        Returns a dictionary of metrics, or None if the ticker cannot be found.
        """
        try:
            self._throttle()
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            
            if not info or 'symbol' not in info:
                logger.warning(f"Ticker {ticker} not found in Yahoo Finance.")
                return None
                
            metrics = {
                "market_cap": info.get("marketCap"),
                "enterprise_value": info.get("enterpriseValue"),
                "revenue": info.get("totalRevenue"),
                "ebitda": info.get("ebitda"),
                "revenue_multiple": info.get("enterpriseToRevenue"),
                "ebitda_multiple": info.get("enterpriseToEbitda"),
                "currency": info.get("financialCurrency", "USD"),
                "period_type": "ttm"  # Yahoo Finance `info` is generally TTM for these fields
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance data for {ticker}: {e}")
            return None
