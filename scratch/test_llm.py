import json
import polars as pl

augmented_rows = [
    {"id": " 0123 ", "name": "Acme", "description": "Tech co", "founders": ["Alice", "Bob"]},
    {"id": "0124", "name": "Beta", "description": None, "founders": []}
]

for row in augmented_rows:
    if "founders" in row and isinstance(row["founders"], list):
        row["founders"] = ", ".join(str(f) for f in row["founders"] if f is not None)

llm_df = pl.DataFrame(augmented_rows)
print("LLM DF:")
print(llm_df)

companies_df = pl.DataFrame([
    {"id": "0123", "name": "Acme", "billingcity": "Hartford"},
    {"id": "0124", "name": "Beta", "billingcity": "New Haven"}
])

companies_df = companies_df.with_columns(pl.col("id").cast(pl.String).str.strip_chars())
llm_df = llm_df.with_columns(pl.col("id").cast(pl.String).str.strip_chars())

if "name" in llm_df.columns:
    llm_df = llm_df.drop("name")

final_df = companies_df.join(llm_df, on="id", how="left")
print("\nFINAL DF:")
print(final_df)
