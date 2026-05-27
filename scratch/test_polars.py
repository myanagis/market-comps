import polars as pl
data = [
    {"a": None},
    {"a": "hello"}
]
try:
    df = pl.DataFrame(data)
    print("SUCCESS")
    print(df.dtypes)
    print(df)
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
