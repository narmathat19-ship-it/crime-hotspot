import pandas as pd
import numpy as np

np.random.seed(42)
n = 10000

df = pd.DataFrame({
    'Date': pd.date_range('2020-01-01', periods=n, freq='h'),
    'Primary Type': np.random.choice(['THEFT','BATTERY','ASSAULT','BURGLARY','ROBBERY'], n),
    'Description': 'SAMPLE',
    'Location Description': 'STREET',
    'Arrest': np.random.choice([True, False], n),
    'District': np.random.randint(1, 25, n),
    'Latitude': np.random.uniform(41.65, 42.05, n),
    'Longitude': np.random.uniform(-87.85, -87.55, n)
})

df.to_csv('data/crimes.csv', index=False)
print('✅ Sample dataset created with', len(df), 'records')