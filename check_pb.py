import pandas as pd

try:
    xl = pd.ExcelFile("Pitchbook N America PE Comps - Apr2026.xlsx")
    print("Sheets:", xl.sheet_names)
    for s in xl.sheet_names:
        print(f"\n--- {s} ---")
        df = xl.parse(s)
        print("Columns:", df.columns.tolist())
        print(df.head(2).to_string())
except Exception as e:
    print(e)
