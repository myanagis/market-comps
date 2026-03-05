from market_comps.waterfall.models import CapTable, SecurityClass, ExitScenario, SecurityType
from market_comps.waterfall.calculator import WaterfallCalculator

print("Building Cap Table...")
ct = CapTable()

# $5m convertible note on $10m post. 
# Converts into 50% of the company at the next round because 5M / 10M cap = 50%.
s1 = SecurityClass(
    series_name="Note",
    security_type=SecurityType.CONVERTIBLE_NOTE,
    total_investment_usd=5000000,
    valuation_cap=10000000
)
ct.add_security(s1)

# $10m raise on $22m post ($12m pre)
s2 = SecurityClass(
    series_name="Series A",
    security_type=SecurityType.PREFERRED,
    total_investment_usd=10000000,
    pre_money_valuation=12000000,
    post_money_valuation=22000000,
    liquidation_preference_multiple=1.0
)
ct.add_security(s2)

# $15M exit
ex = ExitScenario(exit_value_usd=15000000)

print("\nRunning Calculator...")
result = WaterfallCalculator.calculate_exit_waterfall(ct, ex)

print("\n--- CAP TABLE SHARES ---")
for s in ct.securities:
    print(f"{s.series_name}: {s.total_shares} shares")

print("\n--- PAYOUTS ---")
for sec_id, payout in result.payouts.items():
    s = ct.get_security(next(sc.series_name for sc in ct.securities if sc.id == sec_id))
    print(f"{s.series_name}: ${payout:,.2f}")
