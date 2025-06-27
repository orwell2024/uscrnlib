```python
import requests
from bs4 import BeautifulSoup
import pandas as pd
import io
from tqdm import tqdm
from scipy.signal import savgol_filter

# Parameters
rate = '/subhourly01/'
year = 2025
target_day = '2025-06-26'
base_url = f'https://www.ncei.noaa.gov/pub/data/uscrn/products{rate}{year}/'

# Get list of .txt station files
response = requests.get(base_url)
soup = BeautifulSoup(response.content, 'html.parser')
station_files = [link.get('href') for link in soup.find_all('a') if link.get('href', '').endswith('.txt')]

# Optional: limit for quick test
# station_files = station_files[:100]

results = []

for file in tqdm(station_files):
    try:
        url = base_url + file
        r = requests.get(url)
        r.raise_for_status()

        df = pd.read_csv(io.StringIO(r.text), delim_whitespace=True, header=None, names=[
            'WBANNO', 'UTC_DATE', 'UTC_TIME', 'LST_DATE', 'LST_TIME', 'CRX_VN',
            'LONGITUDE', 'LATITUDE', 'AIR_TEMPERATURE', 'PRECIPITATION',
            'SOLAR_RADIATION', 'SR_FLAG', 'SURFACE_TEMPERATURE', 'ST_TYPE',
            'ST_FLAG', 'RELATIVE_HUMIDITY', 'RH_FLAG', 'SOIL_MOISTURE_5',
            'SOIL_TEMPERATURE_5', 'WETNESS', 'WET_FLAG', 'WIND_1_5', 'WIND_FLAG'
        ])

        # Parse datetime and filter
        df['DATETIME'] = pd.to_datetime(df['LST_DATE'].astype(str) + df['LST_TIME'].astype(str).str.zfill(4), format='%Y%m%d%H%M')
        df = df[df['DATETIME'].dt.date == pd.to_datetime(target_day).date()]
        df = df[(df['AIR_TEMPERATURE'] > -90) & (df['AIR_TEMPERATURE'] < 60)]

        if len(df) > 20:
            temps = df['AIR_TEMPERATURE'].interpolate()
            smooth = savgol_filter(temps, window_length=min(21, len(temps)//2*2+1), polyorder=2)
            tmax = float(smooth.max())
            lat = df.iloc[0]['LATITUDE']
            lon = df.iloc[0]['LONGITUDE']
            station_name = file.split('-')[-1].replace('.txt', '')
            results.append({'Station': station_name, 'LAT': lat, 'LON': lon, 'TMAX': tmax})

    except Exception as e:
        print(f"Failed for {file}: {e}")
        continue

# Create DataFrame
df_tmax = pd.DataFrame(results)
df_tmax.to_csv('USCRN_TMAX_2025_06_26.csv', index=False)
df_tmax.head()

```


```python
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from scipy.interpolate import Rbf
import matplotlib.patheffects as pe
from adjustText import adjust_text
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("USCRN_TMAX_2025_06_26.csv")

# Filter for contiguous US (approximate bounding box)
df_contig = df[(df["LON"] > -125) & (df["LON"] < -66) & (df["LAT"] > 24) & (df["LAT"] < 50)].copy()
df_contig["TMAX_F"] = df_contig["TMAX"] * 9 / 5 + 32
df_contig["ShortName"] = df_contig["Station"].str.replace(r'_[0-9]+_[A-Z]+$', '', regex=True)

# Extract max per state (assume state prefix is first 2 letters before first underscore)
df_contig["State"] = df_contig["Station"].str.extract(r"^([A-Z]{2})_")
idx_max_per_state = df_contig.groupby("State")["TMAX_F"].idxmax()
df_max_state = df_contig.loc[idx_max_per_state]

# Grid interpolation for background
grid_x, grid_y = np.mgrid[
    df_contig["LON"].min():df_contig["LON"].max():300j,
    df_contig["LAT"].min():df_contig["LAT"].max():300j
]
rbf = Rbf(df_contig["LON"], df_contig["LAT"], df_contig["TMAX_F"], function='linear')
grid_z = rbf(grid_x, grid_y)

# Begin plotting
fig, ax = plt.subplots(figsize=(16, 10))
m = Basemap(projection='merc',
            llcrnrlon=-125, llcrnrlat=24,
            urcrnrlon=-66, urcrnrlat=50,
            resolution='i', ax=ax)

m.drawcoastlines(color='gray')
m.drawcountries(color='gray')
m.drawstates(color='gray')
m.drawmapboundary(fill_color='#e6f2ff')
m.fillcontinents(color='#f0f0f0', lake_color='#e6f2ff')

# Interpolated field background (without reshape bug)
lon2 = np.linspace(df_contig["LON"].min(), df_contig["LON"].max(), 300)
lat2 = np.linspace(df_contig["LAT"].min(), df_contig["LAT"].max(), 300)
lon2g, lat2g = np.meshgrid(lon2, lat2)

rbf = Rbf(df_contig["LON"], df_contig["LAT"], df_contig["TMAX_F"], function='linear')
grid_z = rbf(lon2g, lat2g)

x2, y2 = m(lon2g, lat2g)
mesh = m.pcolormesh(x2, y2, grid_z, cmap='hot_r', shading='auto', alpha=0.85)

# Label each state’s max T
texts = []
for _, row in df_max_state.iterrows():
    px, py = m(row["LON"], row["LAT"])
    label = f"{row['ShortName'].replace(row['State'] + '_', '')}\n{row['TMAX_F']:.1f}°F"
    text = ax.text(px, py, label, fontsize=14, weight='bold', color='white',
                   ha='center', va='center',
                   bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.4'),
                   path_effects=[pe.withStroke(linewidth=2, foreground="black")])
    texts.append(text)

adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='white', lw=0.5))

# Add colorbar
cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.04)
cbar.set_label("Max Temperature (°F)", fontsize=12)

# Title and footer
ax.set_title("Max Temperature per State – USCRN Stations (June 26, 2025)", fontsize=22, weight='bold')
fig.text(0.5, 0.02,
         "Data: NOAA USCRN | Map: GPT, PA: @sauron2022 | https://www.ncei.noaa.gov/pub/data/uscrn/products/subhourly01/2025/",
         ha='center', fontsize=10, style='italic', color='gray')

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.show()

```



```python

```
![image](https://github.com/user-attachments/assets/68ca8c04-9f64-4a70-9540-4c2e9c831cb1)
