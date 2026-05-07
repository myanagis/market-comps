import sys
sys.path.append('.')
from pages._11_CT_Business_Registry import fetch_recent_ct_businesses, augment_businesses
from market_comps.llm_client import LLMClient
from market_comps.config import MODEL_OPTIONS

df = fetch_recent_ct_businesses('2024-01-01T00:00:00', '2024-02-01T23:59:59', ["5112"])
if df.is_empty():
    print("No data fetched.")
else:
    print("Original cols:", df.columns)
    client = LLMClient(model='openai/gpt-4o-mini')
    final_df, usage = augment_businesses(df, client, max_rows=2)
    print("Final cols:", final_df.columns)
    print("Final dataframe preview:")
    print(final_df.head(2))
