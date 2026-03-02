from typing import Dict, Any, List
from datetime import date
from dateutil.relativedelta import relativedelta
from market_comps.waterfall.models import WaterfallModel, ExitScenario, SecurityType, Security, ExitResult

class WaterfallCalculator:
    """
    Calculates post-money values, ownership percentages, and exit distributions based on a WaterfallModel.
    """
    
    @staticmethod
    def calculate_exit_waterfall(model: WaterfallModel, exit_scenario: ExitScenario) -> ExitResult:
        """
        Main entry point for calculating flow of funds in an exit.
        """
        payouts = {sec.name: 0.0 for sec in model.securities}
        explanations: List[str] = [f"Starting Exit Waterfall for ${exit_scenario.exit_value:,.2f}M on {exit_scenario.exit_date}"]
        remaining_proceeds = exit_scenario.exit_value
        
        if remaining_proceeds <= 0:
            explanations.append("Exit value is 0 or negative. No payouts.")
            return ExitResult(payouts=payouts, explanations=explanations)
            
        # Determine seniority: Backwards by Date (Most recent = Most Senior)
        sorted_securities = sorted(model.securities, key=lambda s: s.date, reverse=True)
        order_names = " -> ".join([s.name for s in sorted_securities])
        explanations.append(f"Processing backwards by seniority (most recent first): {order_names}")
        
        # Calculate fully diluted shares for pro-rata conversions
        total_fd_shares = sum(sec.fully_diluted_shares or sec.total_shares or 0 for sec in model.securities)
        explanations.append(f"Total Fully Diluted Shares: {total_fd_shares:,}")
        
        # Log initial positions
        for sec in sorted_securities:
            sec_shares = sec.fully_diluted_shares or sec.total_shares or 0
            ownership_pct = sec_shares / total_fd_shares if total_fd_shares > 0 else 0.0
            explanations.append(f"[{sec.name}] Initial Position: {sec_shares:,} shares ({ownership_pct * 100:.2f}%)")

        # TODO 2: Convert Notes & SAFEs into Preferred shares prior to distributing preferences.
        for sec in sorted_securities:
            if sec.security_type in [SecurityType.CONVERTIBLE_NOTE, SecurityType.SAFE]:
                invested = sec.capital_raised or 0.0
                
                # 1. Calculate Accrued Interest (if applicable)
                if sec.security_type == SecurityType.CONVERTIBLE_NOTE and sec.interest_rate:
                    rate = sec.interest_rate / 100.0  # Convert 10.0 to 0.10
                    # Calculate duration in years
                    days_held = (exit_scenario.exit_date - sec.date).days
                    years_held = days_held / 365.25
                    
                    if sec.is_compounding:
                        # Compound Interest Formula: P(1+r)^t
                        total_with_interest = invested * ((1 + rate) ** years_held)
                        accrued_interest = total_with_interest - invested
                        explanations.append(f"[{sec.name}] Accrued Compounding Interest: ${invested:,.2f}M \u00d7 (1+{rate*100:.1f}%) ^ {years_held:.2f} yrs = ${accrued_interest:,.2f}M")
                    else:
                        # Simple Interest Formula: P * r * t
                        accrued_interest = invested * rate * years_held
                        total_with_interest = invested + accrued_interest
                        explanations.append(f"[{sec.name}] Accrued Simple Interest: ${invested:,.2f}M \u00d7 {rate*100:.1f}% \u00d7 {years_held:.2f} yrs = ${accrued_interest:,.2f}M")
                        
                    # Temporarily update the security's capital basis for standard payout limits
                    sec.capital_raised = total_with_interest
                # 2. Determine Conversion Price at Exit
                # Since we are converting at Exit, we compare the Valuation Cap (if any) against the Effective Exit Valuation.
                # Estimate a base common share price across the fully diluted pool
                base_share_price = exit_scenario.exit_value / total_fd_shares if total_fd_shares > 0 else 0.0
                
                conversion_price = base_share_price # Default
                
                # Check Discount
                if sec.discount:
                    discount_price = base_share_price * (1 - (sec.discount / 100.0))
                    conversion_price = discount_price
                    explanations.append(f"[{sec.name}] Discounted Price: ${conversion_price:,.4f}/share ({sec.discount}% off implied exit price of ${base_share_price:,.4f})")
                    
                # Check Pre-Money Cap
                if sec.pre_money_cap:
                    # Implied price per share purely under the cap
                    cap_price = sec.pre_money_cap / total_fd_shares if total_fd_shares > 0 else 0.0
                    explanations.append(f"[{sec.name}] Cap Price: ${cap_price:,.4f}/share (Cap: ${sec.pre_money_cap:,.2f}M)")
                    
                    if cap_price < conversion_price:
                        conversion_price = cap_price
                        explanations.append(f"[{sec.name}] Cap price represents the better conversion event.")
                
                if conversion_price > 0:
                    # Generate new shares
                    new_shares = int(invested / conversion_price)
                    sec.fully_diluted_shares = new_shares
                    # Add newly minted shares to the global pool for downstream calculations
                    total_fd_shares += new_shares
                    
                    explanations.append(f"[{sec.name}] Converted ${invested:,.2f}M to {new_shares:,} Common Shares at ${conversion_price:,.4f}/share.")
                else:
                    explanations.append(f"[{sec.name}] Conversion price is 0. No shares generated.")
                
        # TODO 3: Payout Liquidation Preferences (Looping from most to least senior)
        remaining_fd_shares = total_fd_shares
        
        for sec in sorted_securities:
            if remaining_proceeds <= 0:
                explanations.append("Proceeds exhausted.")
                break
                
            explanations.append(f"[{sec.name}] Waterfall Step: ${remaining_proceeds:,.2f}M remaining | {remaining_fd_shares:,} FD shares remaining")
                
            if sec.security_type == SecurityType.PREFERRED:
                # Calculate Ownership % based on remaining shares
                sec_shares = sec.fully_diluted_shares or sec.total_shares or 0
                ownership_pct = sec_shares / remaining_fd_shares if remaining_fd_shares > 0 else 0.0
                explanations.append(f"[{sec.name}] Shares: {sec_shares:,} | Remaining Fully-Diluted Shares: {remaining_fd_shares:,} | Dynamic Ownership: {ownership_pct * 100:.2f}%")

                # 1. Calculate Liquidation Preference
                liq_mult = getattr(sec, 'liquidity_preference', 1.0)
                invested = sec.capital_raised or 0.0
                liq_pref = invested * liq_mult
                explanations.append(f"[{sec.name}] Liquidation Preference Calculation: ${invested:,.2f}M raised \u00d7 {liq_mult}x = ${liq_pref:,.2f}M")
                
                # 2. Calculate As-Converted value
                if remaining_fd_shares > 0:
                    as_converted_value = remaining_proceeds * ownership_pct
                    explanations.append(f"[{sec.name}] As-Converted Calculation: ${remaining_proceeds:,.2f}M remaining \u00d7 {ownership_pct * 100:.2f}% ownership = ${as_converted_value:,.2f}M")
                else:
                    as_converted_value = 0.0
                    explanations.append(f"[{sec.name}] As-Converted Calculation: $0.00M (0% ownership)")
                    
                explanations.append(f"[{sec.name}] Comparing Liq Pref (${liq_pref:,.2f}M) vs As-Converted (${as_converted_value:,.2f}M)")
                
                if as_converted_value > liq_pref:
                    payout_amount = min(as_converted_value, remaining_proceeds)
                    limiter_note = " (Limited by remaining proceeds)" if remaining_proceeds < as_converted_value else ""
                    explanations.append(f"[{sec.name}] Investor chose to convert to common. Payout: ${payout_amount:,.2f}M{limiter_note}")
                    # If they convert, their shares remain in the pool to share remaining proceeds pro-rata (if participating, or if this is final step)
                    # Note: Standard non-participating preferred takes either LiqPref OR As-Converted, not both. 
                    # If they take As-Converted, they effectively 'ate' their portion of the total proceeds directly.
                    # To model this properly in a step-by-step waterfall, taking As-Converted removes them and their shares from future steps
                    remaining_fd_shares -= sec_shares
                else:
                    payout_amount = min(liq_pref, remaining_proceeds)
                    limiter_note = " (Limited by remaining proceeds)" if remaining_proceeds < liq_pref else ""
                    explanations.append(f"[{sec.name}] Investor took liquidation preference. Payout: ${payout_amount:,.2f}M{limiter_note}")
                    # If they take LiqPref (and are non-participating), their shares are dead for the rest of the waterfall
                    remaining_fd_shares -= sec_shares
                    
                payouts[sec.name] = payout_amount
                remaining_proceeds -= payout_amount
                
            elif sec.security_type in [SecurityType.SAFE, SecurityType.CONVERTIBLE_NOTE]:
                # If they converted to shares in TODO 2, they act exactly like common/as-converted equity here.
                # Do they have a liquidation preference (e.g., getting their money back 1x instead of converting)?
                sec_shares = sec.fully_diluted_shares or 0
                ownership_pct = sec_shares / remaining_fd_shares if remaining_fd_shares > 0 else 0.0
                
                invested = sec.capital_raised or 0.0
                liq_pref = invested * getattr(sec, 'liquidity_preference', 1.0) # Usually 1x for Notes
                
                as_converted_value = remaining_proceeds * ownership_pct if remaining_fd_shares > 0 else 0.0
                
                explanations.append(f"[{sec.name}] Comparing Liq Pref (${liq_pref:,.2f}M) vs Converted Equity Value (${as_converted_value:,.2f}M)")
                
                if as_converted_value > liq_pref:
                    payout_amount = min(as_converted_value, remaining_proceeds)
                    limiter_note = " (Limited by remaining proceeds)" if remaining_proceeds < as_converted_value else ""
                    explanations.append(f"[{sec.name}] Noteholder chose to convert. Payout: ${payout_amount:,.2f}M{limiter_note}")
                    # Shares removed for future steps as they took their equity payout now
                    remaining_fd_shares -= sec_shares 
                else:
                    payout_amount = min(liq_pref, remaining_proceeds)
                    limiter_note = " (Limited by remaining proceeds)" if remaining_proceeds < liq_pref else ""
                    explanations.append(f"[{sec.name}] Noteholder took return of capital (1x Liq Pref). Payout: ${payout_amount:,.2f}M{limiter_note}")
                    # Their converted shares are annulled, removed from the remaining pool
                    remaining_fd_shares -= sec_shares
                    
                payouts[sec.name] = payout_amount
                remaining_proceeds -= payout_amount
                
        # TODO 4: Distribute remaining proceeds pro-rata to Common & Participating Preferred
        if remaining_proceeds > 0:
            explanations.append(f"[Common] Distributing remaining ${remaining_proceeds:,.2f}M proceeds pro-rata to {remaining_fd_shares:,} remaining Common shares")
            
            # Simple pro-rata for common shares initially
            if remaining_fd_shares > 0:
                common_deal_price = remaining_proceeds / remaining_fd_shares
                
                for sec in model.securities:
                    if sec.security_type == SecurityType.COMMON:
                        sec_shares = sec.fully_diluted_shares or sec.total_shares or 0
                        ownership_pct = sec_shares / remaining_fd_shares if remaining_fd_shares > 0 else 0.0
                        explanations.append(f"[{sec.name}] Shares: {sec_shares:,} | Dynamic Ownership: {ownership_pct * 100:.2f}%")

                        payout = sec_shares * common_deal_price
                        payouts[sec.name] = payouts.get(sec.name, 0.0) + payout
                        remaining_proceeds -= payout
                        explanations.append(f"[{sec.name}] Common pro-rata payout: ${payout:,.2f}M at ${common_deal_price:,.6f}/share")
            else:
                explanations.append("[Common] Error: No fully diluted shares found to distribute remaining proceeds.")
                
        explanations.append("Waterfall calculation complete.")
        
        # Calculate MOIC
        moic = {}
        for sec in model.securities:
            invested = sec.capital_raised or 0.0
            if invested > 0:
                moic[sec.name] = payouts.get(sec.name, 0.0) / invested
            else:
                moic[sec.name] = 0.0 # N/A or Infinite for common/founders with 0 basis
        
        return ExitResult(payouts=payouts, moic=moic, explanations=explanations)
