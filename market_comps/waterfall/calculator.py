from typing import Dict, Any, List
import datetime
from market_comps.waterfall.models import CapTable, ExitScenario, SecurityType, SecurityClass

def _days_between(d1_str: str, d2_str: str) -> int:
    try:
        d1 = datetime.datetime.strptime(d1_str, "%Y-%m-%d")
        d2 = datetime.datetime.strptime(d2_str, "%Y-%m-%d")
        return max(0, (d2 - d1).days)
    except:
        return 365 # Default 1 year if parsing fails

class WaterfallCalculator:
    """
    Calculates post-money values, ownership percentages, and exit distributions based on a CapTable.
    """
    
    @staticmethod
    def calculate_exit_waterfall(cap_table: CapTable, exit_scenario: ExitScenario) -> ExitScenario:
        result_scenario = exit_scenario.model_copy()
        result_scenario.payouts = {sec.id: 0.0 for sec in cap_table.securities}
        
        available_funds = exit_scenario.exit_value_usd
        if available_funds <= 0:
            return result_scenario
            
        common_shares = sum(s.total_shares or 0 for s in cap_table.securities if s.security_type == SecurityType.COMMON)
        if common_shares == 0:
            common_shares = 1_000_000 # fallback to prevent div by zero
            
        # 1. Evaluate Accrued Interest for Notes
        accrued_values: Dict[str, float] = {}
        for sec in cap_table.securities:
            base_amt = sec.total_investment_usd or 0.0
            accrued = base_amt
            if sec.security_type == SecurityType.CONVERTIBLE_NOTE and sec.interest_rate:
                days = 365
                if sec.close_date and exit_scenario.exit_date:
                    days = _days_between(sec.close_date, exit_scenario.exit_date)
                years = days / 365.0
                
                if sec.is_interest_compounding:
                    accrued = base_amt * ((1 + sec.interest_rate) ** years)
                else:
                    accrued = base_amt + (base_amt * sec.interest_rate * years)
            accrued_values[sec.id] = accrued

        # 2. Distribute Liquidation Preferences (Seniority order)
        # Assuming lower number is more senior
        sorted_prefs = sorted([s for s in cap_table.securities if s.security_type == SecurityType.PREFERRED], key=lambda x: x.seniority)
        
        pref_payouts: Dict[str, float] = {sec.id: 0.0 for sec in sorted_prefs}
        
        for p_sec in sorted_prefs:
            liq_pref = (p_sec.total_investment_usd or 0.0) * p_sec.liquidation_preference_multiple
            payout = min(liq_pref, available_funds)
            pref_payouts[p_sec.id] = payout
            available_funds -= payout
            result_scenario.payouts[p_sec.id] += payout
            if available_funds <= 0:
                break
                
        # 3. Simulate conversion of SAFEs / Notes and decide if Preferred converts
        # We need to find the Common Share Price. 
        # But SAFE conversion depends on Common Share Price, and Common Share Price depends on SAFE conversion.
        # We simplify: SAFE Valuation Cap Price = Cap / (Common Shares)
        
        converting_shares_total = 0.0
        conversion_shares: Dict[str, float] = {}
        
        # Calculate SAFE/Note shares
        for sec in cap_table.securities:
            if sec.security_type in [SecurityType.SAFE, SecurityType.CONVERTIBLE_NOTE]:
                accrued = accrued_values[sec.id]
                cap_price = (sec.valuation_cap / common_shares) if sec.valuation_cap else float('inf')
                # Wait, "discount to deal price" requires solving for deal price.
                # If we don't have deal price yet, we can't accurately check the discount.
                # Since Streamlit allows iterative numerical loops, let's just use cap price if available.
                # If no cap price, we assign them a dummy share count and iterate, or we just use cap.
                # For this MVP, we use the Valuation Cap price if it exists, otherwise assume 0 shares till we know deal price.
                shrs = accrued / cap_price if cap_price != float('inf') else 0
                conversion_shares[sec.id] = shrs
                converting_shares_total += shrs
            elif sec.security_type == SecurityType.PREFERRED:
                # Preferred base shares
                conversion_shares[sec.id] = sec.total_shares or 0.0
                
        # Test Common Share Price if everyone remaining converts
        # (Remaining proceeds / Participating & Common & Converted shares)
        # Standard non-participating preferred converts IF their common payout > their Liq Pref payout.
        
        # Let's do a simple iterative loop to find the clearing price
        deal_price = 0.0
        for _ in range(10): # 10 iterations is usually enough to converge
            participating_shares = common_shares
            
            # Recalculate SAFE shares using discount if it's better than cap
            for sec in cap_table.securities:
                if sec.security_type in [SecurityType.SAFE, SecurityType.CONVERTIBLE_NOTE]:
                    accrued = accrued_values[sec.id]
                    cap_price = (sec.valuation_cap / common_shares) if sec.valuation_cap else float('inf')
                    discount_price = deal_price * (1.0 - (sec.discount_rate or 0.0)) if deal_price > 0 else float('inf')
                    
                    best_price = min(cap_price, discount_price)
                    if best_price == float('inf') or best_price <= 0:
                        conversion_shares[sec.id] = 0
                    else:
                        conversion_shares[sec.id] = accrued / best_price
                    participating_shares += conversion_shares[sec.id]
                        
            # Who participates in the residual?
            for sec in sorted_prefs:
                # Non-participating Preferred ONLY participates if they convert (forfeit Liq Pref)
                if not sec.is_participating:
                    as_converted_value = conversion_shares[sec.id] * deal_price
                    if as_converted_value > pref_payouts[sec.id]:
                        # They convert
                        participating_shares += conversion_shares[sec.id]
                else:
                    # Participating Preferred always participates in residual (up to cap)
                    participating_shares += conversion_shares[sec.id]
                    
            if participating_shares > 0:
                deal_price = available_funds / participating_shares
            else:
                deal_price = 0.0
                
        # 4. Final Distribution of Residuals
        # Apply the final converged deal price
        # Note: True participating caps require piecewise distribution. 
        # For MVP, we'll cap their total payout strictly at the end.
        
        remaining_funds = available_funds
        
        for sec in cap_table.securities:
            if remaining_funds <= 0:
                break
                
            if sec.security_type == SecurityType.COMMON:
                payout = common_shares * deal_price
                result_scenario.payouts[sec.id] += payout
                remaining_funds -= payout
                
            elif sec.security_type in [SecurityType.SAFE, SecurityType.CONVERTIBLE_NOTE]:
                payout = conversion_shares[sec.id] * deal_price
                result_scenario.payouts[sec.id] += payout
                remaining_funds -= payout
                
            elif sec.security_type == SecurityType.PREFERRED:
                if not sec.is_participating:
                    as_convert = conversion_shares[sec.id] * deal_price
                    if as_convert > pref_payouts[sec.id]:
                        # Converted: Take as_convert, give back Liq Pref
                        result_scenario.payouts[sec.id] = as_convert
                        remaining_funds -= (as_convert - pref_payouts[sec.id]) # Net diff
                else:
                    # Participating: They get Liq Pref + Pro Rata Common
                    pro_rata = conversion_shares[sec.id] * deal_price
                    cap = sec.participation_cap_multiple * (sec.total_investment_usd or 0.0) if sec.participation_cap_multiple else float('inf')
                    
                    total_potential = pref_payouts[sec.id] + pro_rata
                    actual_total = min(total_potential, cap)
                    
                    additional_payout = actual_total - pref_payouts[sec.id]
                    result_scenario.payouts[sec.id] += additional_payout
                    remaining_funds -= additional_payout
                    
        return result_scenario
