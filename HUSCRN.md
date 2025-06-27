```python

# Educational Notebook : USCRN Legacy Pairs, GHCNv4 RAW. 
# 8 golden USCRN pairs used
import requests
import tarfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import io

# === Settings ===
baseline_start = 1951
baseline_end = 1980
min_months = 11
start_year = 1880
end_year = 2024
smoothing_frac = 0.13

# === Step 1: Download and extract
url = "https://www.ncei.noaa.gov/pub/data/ghcn/v4/ghcnm.tavg.latest.qcu.tar.gz"
response = requests.get(url, stream=True)
tar_bytes = io.BytesIO(response.content)

# Station pairs, the magnificent 8 USCRN (HUSCRN) with more than 90 years data.
pairs = [
    ("USC00012813", "USW00063869"),
    ("USC00294426", "USW00003074"),
    ("USC00402202", "USW00063855"),
    ("USC00250030", "USW00094077"),
    ("USC00018385", "USW00073801"),
    ("USC00380764", "USW00063826"),
    ("USC00013160", "USW00063892"),
    ("USC00348501", "USW00053926")
]
station_ids = {sid for p in pairs for sid in p}

# === Step 2: Extract and grep lines
filtered_lines = []
with tarfile.open(fileobj=tar_bytes, mode="r:gz") as tar:
    dat_member = next(m for m in tar.getmembers() if m.name.endswith(".qcu.dat"))
    dat_file = tar.extractfile(dat_member)
    for line in dat_file:
        sid = line[0:11].decode("utf-8").strip()
        if sid in station_ids and line[15:19].decode("utf-8") == "TAVG":
            filtered_lines.append(line.decode("utf-8"))

# === Step 3: Parse temperature data
records = []
for line in filtered_lines:
    sid = line[0:11].strip()
    year = int(line[11:15])
    if year < start_year or year > end_year:
        continue
    monthly = [int(line[19 + m*8:24 + m*8]) for m in range(12)]
    monthly = [v / 100.0 if v != -9999 else None for v in monthly]
    if sum(v is not None for v in monthly) >= min_months:
        avg = np.mean([v for v in monthly if v is not None])
        records.append([year, avg, sid])

df = pd.DataFrame(records, columns=["year", "tavg", "station_id"])

# === Step 4: Build per-pair anomalies (with pairwise baseline subtraction)
pairwise_anomalies = []

for legacy_id, uscrn_id in pairs:
    df_legacy = df[df["station_id"] == legacy_id].set_index("year")
    df_uscrn = df[df["station_id"] == uscrn_id].set_index("year")
    all_years = sorted(set(df_legacy.index).union(df_uscrn.index))
    combined = []
    for y in all_years:
        temps = []
        if y in df_legacy.index:
            temps.append(df_legacy.loc[y, "tavg"])
        if y in df_uscrn.index:
            temps.append(df_uscrn.loc[y, "tavg"])
        if temps:
            combined.append([y, np.mean(temps)])
    df_pair = pd.DataFrame(combined, columns=["year", "tavg"])
    
    # Compute pair-specific baseline
    base = df_pair[(df_pair["year"] >= baseline_start) & (df_pair["year"] <= baseline_end)]
    if not base.empty:
        baseline_mean = base["tavg"].mean()
        df_pair["anomaly"] = df_pair["tavg"] - baseline_mean
        pairwise_anomalies.append(df_pair[["year", "anomaly"]])

# === Step 5: Aggregate anomalies across all pairs (after baseline subtraction)
df_all = pd.concat(pairwise_anomalies)
df_agg = df_all.groupby("year")["anomaly"].mean().reset_index()

# === Step 6: LOESS smoothing
loess = sm.nonparametric.lowess(endog=df_agg["anomaly"],
                                exog=df_agg["year"], frac=smoothing_frac)

# === Step 7: Plot (dark background, styled)
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df_agg["year"], df_agg["anomaly"], label="Aggregated Anomaly", color='cyan',
        linewidth=2, marker='o', markersize=4, alpha=0.9)
ax.plot(loess[:, 0], loess[:, 1], color='cyan', linewidth=4, alpha=0.6, label="LOESS Trend")

ax.set_title("Historic USCRN Temperature Anomalies (Baseline 1920–1940) – GHCN RAW", fontsize=15, weight='bold')
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Temperature Anomaly (°C)", fontsize=12)
ax.set_ylim(-2, 3)
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left", fontsize=10)

fig.text(0.01, -0.12,
         "Data Source: GHCN v4.0.1 QCU – NOAA\n"
         "Yearly averages computed from >=11 months per year.\n"
         "Anomalies computed per station pair using 1920–1940 baseline.\n"
         "Then averaged across all 8 pairs.\n",
         fontsize=9, color='white')

plt.tight_layout()
plt.show()

# === Step 7: Plot – NYT-style
plt.style.use('default')
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_facecolor("white")
fig.patch.set_facecolor("white")

# Raw data points as small gray dots
ax.scatter(df_agg["year"], df_agg["anomaly"], color='gray', alpha=0.3, s=10)

# Thick LOESS smoothed trend line
ax.plot(loess[:, 0], loess[:, 1], color='black', linewidth=3)

# Aesthetic refinements
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlim(1880, 2024)
ax.set_ylim(-1, 2)
ax.set_title("Historic USCRN Temperature Anomalies", fontsize=16, weight='bold')
ax.set_ylabel("Temperature anomaly (°C)")
ax.set_xlabel("")
ax.tick_params(axis='both', which='major', labelsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Subtle contextual labels
ax.text(1885, 1.6, "Hotter than average", fontsize=10, color="gray")
ax.text(1885, -0.4, "Cooler than average", fontsize=10, color="gray")

# Caption
fig.text(0.01, -0.05,
         "Source: GHCN v4 QCU, legacy USCRN pairs · Anomalies relative to 1951–1980 baseline.\n"
         "Yearly averages with >=11 valid months. Reproduced in NYT visual style.",
         fontsize=9, color='black')

plt.tight_layout()
plt.show()
```python
```
![image](https://github.com/user-attachments/assets/24035c63-0a64-4365-90cd-62f542588fc7)



