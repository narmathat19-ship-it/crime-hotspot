import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

# Indian state coordinates
STATE_COORDS = {
    'Tamil Nadu': (11.1271, 78.6569),
    'Andhra Pradesh': (15.9129, 79.7400),
    'Kerala': (10.8505, 76.2711),
    'Karnataka': (15.3173, 75.7139),
    'Maharashtra': (19.7515, 75.7139),
    'Delhi': (28.7041, 77.1025),
    'Uttar Pradesh': (26.8467, 80.9462),
    'West Bengal': (22.9868, 87.8550),
    'Rajasthan': (27.0238, 74.2179),
    'Gujarat': (22.2587, 71.1924),
    'Madhya Pradesh': (22.9734, 78.6569),
    'Bihar': (25.0961, 85.3131),
    'Odisha': (20.9517, 85.0985),
    'Punjab': (31.1471, 75.3412),
    'Haryana': (29.0588, 76.0856),
    'Telangana': (18.1124, 79.0193),
    'Assam': (26.2006, 92.9376),
    'Jharkhand': (23.6102, 85.2799),
    'Uttarakhand': (30.0668, 79.0193),
    'Himachal Pradesh': (31.1048, 77.1734),
    'Goa': (15.2993, 74.1240),
    'Tripura': (23.9408, 91.9882),
    'Manipur': (24.6637, 93.9063),
    'Meghalaya': (25.4670, 91.3662),
    'Nagaland': (26.1584, 94.5624),
    'Arunachal Pradesh': (28.2180, 94.7278),
    'Mizoram': (23.1645, 92.9376),
    'Sikkim': (27.5330, 88.5122),
    'Andaman & Nicobar Islands': (11.7401, 92.6586),
    'Chandigarh': (30.7333, 76.7794),
    'Puducherry': (11.9416, 79.8083),
    'Jammu & Kashmir': (33.7782, 76.5762),
    'Chhattisgarh': (21.2787, 81.8661),
}

def load_and_clean(filepath):
    df = pd.read_csv(filepath)

    # Add coordinates based on state name
    df['Latitude']  = df['Area_Name'].map(lambda x: STATE_COORDS.get(x, (20.5937, 78.9629))[0])
    df['Longitude'] = df['Area_Name'].map(lambda x: STATE_COORDS.get(x, (20.5937, 78.9629))[1])

    # Add noise so points don't overlap
    df['Latitude']  += np.random.uniform(-1.5, 1.5, len(df))
    df['Longitude'] += np.random.uniform(-1.5, 1.5, len(df))

    # Use Group_Name as crime type
    df['Primary Type'] = df['Group_Name']
    df['Arrest'] = True

    # Add time features
    df['Hour']  = np.random.randint(0, 24, len(df))
    df['Day']   = np.random.randint(0, 7,  len(df))
    df['Month'] = np.random.randint(1, 13, len(df))

    df = df.dropna(subset=['Latitude', 'Longitude', 'Primary Type'])

    print(f"✅ Loaded {len(df)} records")
    return df


def run_clustering(df):
    coords = df[['Latitude', 'Longitude']].values
    kms_per_radian = 6371.0088
    epsilon = 3.0 / kms_per_radian

    db = DBSCAN(
        eps=epsilon,
        min_samples=5,
        algorithm='ball_tree',
        metric='haversine'
    ).fit(np.radians(coords))

    df['Cluster'] = db.labels_
    n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    print(f"✅ Found {n_clusters} crime hotspot clusters")
    return df


def train_model(df):
    le = LabelEncoder()
    df['CrimeEncoded'] = le.fit_transform(df['Primary Type'])

    features = ['Hour', 'Day', 'Month', 'Latitude', 'Longitude']
    X = df[features]
    y = df['CrimeEncoded']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f"✅ Model Accuracy: {acc:.2%}")

    pickle.dump(model, open('model.pkl', 'wb'))
    pickle.dump(le,    open('encoder.pkl', 'wb'))

    return model, le, acc


def predict_crime(hour, day, month, lat, lon):
    model = pickle.load(open('model.pkl', 'rb'))
    le    = pickle.load(open('encoder.pkl', 'rb'))
    input_data = np.array([[hour, day, month, lat, lon]])
    prediction = model.predict(input_data)
    return le.inverse_transform(prediction)[0]