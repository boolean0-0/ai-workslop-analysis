from pathlib import Path
import pandas as pd
import numpy as np

META_PATH = Path('data/raw/posts_meta.parquet')
df = pd.read_parquet(META_PATH)

df['published'] = pd.to_datetime(df['published'], utc=True)
df['month'] = df['published'].dt.tz_convert(None).dt.to_period('M').astype(str)

by_pub = df.groupby('pub').size().sort_values(ascending=False)
print('Counts by pub:')
print(by_pub.to_string())

mc = df.groupby(['pub','month']).size().unstack(fill_value=0)
std_per_pub = mc.std(axis=1)
min_per_pub = mc.min(axis=1)
max_per_pub = mc.max(axis=1)

print('\nStd of monthly counts (lower is better):')
print(std_per_pub.sort_values().to_string())

print('\nMin/Max monthly counts per pub (last 24 months columns if available):')
summary = pd.DataFrame({
    'min_monthly': min_per_pub,
    'max_monthly': max_per_pub,
    'std_monthly': std_per_pub,
    'total': by_pub
}).sort_values('total', ascending=False)

# Add min/max article dates per pub
date_range = df.groupby('pub')['published'].agg(['min','max'])
if not date_range.empty:
    # Convert to naive date strings for readability
    date_range['min'] = date_range['min'].dt.tz_convert(None)
    date_range['max'] = date_range['max'].dt.tz_convert(None)
    date_range = date_range.rename(columns={'min': 'min_date', 'max': 'max_date'})
    date_range['min_date'] = date_range['min_date'].dt.strftime('%Y-%m-%d')
    date_range['max_date'] = date_range['max_date'].dt.strftime('%Y-%m-%d')
    summary = summary.join(date_range[['min_date','max_date']], how='left')

print(summary.to_string())


