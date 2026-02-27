from market_comps.waterfall.models import CapTable, SecurityClass, SecurityType, ExitScenario
from market_comps.waterfall.calculator import WaterfallCalculator

ct = CapTable()
ct.add_security(SecurityClass(
    series_name="Common Founders",
    security_type=SecurityType.COMMON,
    total_shares=10000000
))
ct.add_security(SecurityClass(
    series_name="Seed SAFE",
    security_type=SecurityType.SAFE,
    valuation_cap=15_000_000,
    total_investment_usd=3_000_000
))
ct.add_security(SecurityClass(
    series_name="Series A",
    security_type=SecurityType.PREFERRED,
    seniority=1,
    liquidation_preference_multiple=1.0,
    total_investment_usd=10_000_000,
    total_shares=3000000,
    is_participating=False
))

print("Cap Table initialized.")

exit_low = ExitScenario(exit_value_usd=15_000_000)
exit_mid = ExitScenario(exit_value_usd=50_000_000)
exit_high = ExitScenario(exit_value_usd=100_000_000)

calc = WaterfallCalculator()

for exit_scen in [exit_low, exit_mid, exit_high]:
    res = calc.calculate_exit_waterfall(ct, exit_scen)
    print(f"\n--- EXIT: ${exit_scen.exit_value_usd:,.0f} ---")
    for sec_id, payout in res.payouts.items():
        sec = ct.get_security([s for s in ct.securities if s.id == sec_id][0].series_name)
        print(f"{sec.series_name}: ${payout:,.2f}")
