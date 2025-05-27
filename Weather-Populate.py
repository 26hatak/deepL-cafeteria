
import pandas as pd
import requests
from io import StringIO

# ── CONFIG ─────────────────────────────────────────────────────────────────────
API_KEY    = "///////"
LAT, LON   = 35.681662937151174, 139.521698252871
INPUT_CSV  = "daily_sort.csv"
OUTPUT_CSV = "enriched.csv"
BASE_URL   = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"

# ── HELPER ─────────────────────────────────────────────────────────────────────
def fetch_day_data(date_str):
    """
    Fetch daily weather for a single date_str (YYYY-MM-DD).
    Logs the URL and first 200 chars of the response if no data.
    """
    # By specifying both start and end as the same date,
    # you force the API to return *only* that day.
    url = (
        f"{BASE_URL}{LAT},{LON}/{date_str}/{date_str}"
        f"?unitGroup=metric"
        f"&include=days"
        f"&contentType=csv"
        f"&key={API_KEY}"
    )
    resp = requests.get(url)
    if resp.status_code != 200:
        print(f"⚠️  Non-200 from VC for {date_str}: {resp.status_code}")
        print("Response body:", resp.text[:200], "…")
        resp.raise_for_status()
    # Parse CSV
    df_csv = pd.read_csv(StringIO(resp.text))
    if df_csv.empty:
        print(f"⚠️  No rows returned for {date_str}! Response:\n{resp.text[:200]}…")
        return None
    row = df_csv.iloc[0]
    return {
        "temp_c":       row["temp"],
        "temp_min_c":   row["tempmin"],
        "temp_max_c":   row["tempmax"],
        "feelslike_c":  row["feelslike"],
        "humidity_pct": row["humidity"],
        "precip_mm":    row.get("precip", 0.0)
    }

# ── MAIN ───────────────────────────────────────────────────────────────────────
# 1. Load CSV (must have 'date' column)
df = pd.read_csv(INPUT_CSV, parse_dates=["date"])
df["ds"] = df["date"].dt.strftime("%Y-%m-%d")

# 2. Fetch & cache daily data
cache = {}
for d in df["ds"].unique():
    cache[d] = fetch_day_data(d)

# 3. Map back into DataFrame
weather_df = pd.DataFrame([cache[d] for d in df["ds"]], index=df.index)
df = pd.concat([df, weather_df], axis=1)

# 4. Save enriched CSV
df.drop(columns=["ds"], inplace=True)
df.to_csv(OUTPUT_CSV, index=False)
print(f"Enriched data saved to {OUTPUT_CSV}")
