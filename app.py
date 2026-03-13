from flask import Flask, render_template, request, jsonify
import pandas as pd
import folium
from folium.plugins import HeatMap
import json
from model import load_and_clean, run_clustering, train_model, predict_crime

app = Flask(__name__)

# Load data once on startup
print("Loading data...")
df = load_and_clean('data/32_Murder_victim_age_sex.csv')
df = run_clustering(df)
model, le, accuracy = train_model(df)
print("✅ App ready!")


@app.route('/')
def index():
    # Summary stats for home page
    stats = {
        'total_crimes': len(df),
        'top_crime': df['Primary Type'].value_counts().index[0],
        'peak_hour': int(df['Hour'].value_counts().index[0]),
        'accuracy': f"{accuracy:.2%}",
        'total_clusters': int(df['Cluster'].max()) + 1
    }
    return render_template('index.html', stats=stats)


@app.route('/map')
def map_view():
    # Generate folium heatmap
    m = folium.Map(
        location=[20.5937, 78.9629],
        zoom_start=5,
        tiles='CartoDB dark_matter'  # dark theme looks great!
    )

    # Add heatmap layer
    heat_data = df[['Latitude', 'Longitude']].dropna().values.tolist()
    HeatMap(
        heat_data,
        radius=8,
        blur=10,
        gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 1.0: 'red'}
    ).add_to(m)

    # Add cluster center markers
    cluster_centers = df[df['Cluster'] != -1].groupby('Cluster')[
        ['Latitude', 'Longitude']
    ].mean()

    for idx, row in cluster_centers.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=8,
            color='red',
            fill=True,
            popup=f"Hotspot #{idx}"
        ).add_to(m)

    m.save('static/map.html')
    return render_template('map.html')


@app.route('/dashboard')
def dashboard():
    # Crime type distribution
    crime_counts = df['Primary Type'].value_counts().head(10)
    hourly = df['Hour'].value_counts().sort_index()
    daily  = df['Day'].value_counts().sort_index()

    return render_template('dashboard.html',
        crime_labels=crime_counts.index.tolist(),
        crime_values=crime_counts.values.tolist(),
        hour_labels=hourly.index.tolist(),
        hour_values=hourly.values.tolist(),
        daily_labels=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'],
        daily_values=daily.values.tolist()
    )


@app.route('/predict', methods=['POST'])
def predict():
    data  = request.json
    hour  = int(data['hour'])
    day   = int(data['day'])
    month = int(data['month'])
    lat   = float(data['lat'])
    lon   = float(data['lon'])

    # Predict crime type
    result = predict_crime(hour, day, month, lat, lon)

    # Find nearby crimes (within ~50km radius)
    df['distance'] = ((df['Latitude'] - lat)**2 + 
                      (df['Longitude'] - lon)**2)**0.5
    nearby = df[df['distance'] < 1.0]  # ~50km radius

    if len(nearby) > 0:
        top_crime    = nearby['Primary Type'].value_counts().index[0]
        total_crimes = len(nearby)
        peak_hour    = int(nearby['Hour'].value_counts().index[0])
        peak_month   = int(nearby['Month'].value_counts().index[0])

        # Crime breakdown (top 3)
        breakdown = nearby['Primary Type'].value_counts().head(3)
        breakdown_dict = breakdown.to_dict()

        # Risk level
        if total_crimes > 100:
            risk = '🔴 HIGH'
        elif total_crimes > 50:
            risk = '🟡 MEDIUM'
        else:
            risk = '🟢 LOW'

        # Month name
        month_names = ['', 'January', 'February', 'March', 'April',
                       'May', 'June', 'July', 'August', 'September',
                       'October', 'November', 'December']

        summary = {
            'total_crimes'  : total_crimes,
            'top_crime'     : top_crime,
            'peak_hour'     : f"{peak_hour}:00",
            'peak_month'    : month_names[peak_month],
            'risk_level'    : risk,
            'breakdown'     : breakdown_dict
        }
    else:
        summary = {
            'total_crimes'  : 0,
            'top_crime'     : 'No data',
            'peak_hour'     : 'N/A',
            'peak_month'    : 'N/A',
            'risk_level'    : '🟢 LOW',
            'breakdown'     : {}
        }

    return jsonify({
        'predicted_crime': result,
        'summary'        : summary
    })
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)