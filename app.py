from flask import Flask, render_template, request, jsonify
import requests
import datetime
import joblib
import os
import pandas as pd
import numpy as np

# Added these two imports to fix the Render/Gunicorn bug!
import sys
import __main__

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class RF_DT_SVM_Cascade:
    def __init__(self):
        self.stage1 = RandomForestClassifier(
            n_estimators=400, max_depth=None, random_state=42, n_jobs=-1, class_weight='balanced'
        )
        self.stage2 = DecisionTreeClassifier(
            max_depth=None, min_samples_split=5, min_samples_leaf=2,
            random_state=42, class_weight='balanced'
        )
        self.stage3 = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='rbf', probability=True, C=1000, gamma='scale', random_state=42))
        ])

    def fit(self, X, y):
        self.stage1.fit(X, y)
        s1_probs = self.stage1.predict_proba(X)
        X_s2 = np.column_stack((X, s1_probs))
        
        self.stage2.fit(X_s2, y)
        s2_probs = self.stage2.predict_proba(X_s2)
        X_s3 = np.column_stack((X, s2_probs))
        
        self.stage3.fit(X_s3, y)

    def predict(self, X):
        s1_probs = self.stage1.predict_proba(X)
        X_s2 = np.column_stack((X, s1_probs))
        s2_probs = self.stage2.predict_proba(X_s2)
        X_s3 = np.column_stack((X, s2_probs))
        return self.stage3.predict(X_s3)

# =====================================================================
# THE RENDER FIX: Copy the class into the __main__ namespace for joblib
# =====================================================================
setattr(sys.modules['__main__'], 'RF_DT_SVM_Cascade', RF_DT_SVM_Cascade)
__main__.RF_DT_SVM_Cascade = RF_DT_SVM_Cascade
# =====================================================================

app = Flask(__name__)

model_path = 'crop_model.pkl'
model = None
le = None
poly = None

if os.path.exists(model_path):
    try:
        artifacts = joblib.load(model_path)
        model = artifacts["model"]
        le = artifacts["encoder"]
        poly = artifacts["poly"]
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
else:
    print(f"Could not find {model_path}")

sensor_data = {
    "temp": 0,
    "humidity": 0,
    "ph": 6.5,
    "soil_moisture": 0,
    "seasonal_rain": 0.0,
    "last_lat": 0.0,
    "last_lon": 0.0,
    "address": "Locating...",
    "land_type": "Analyzing..."
}

CROP_NEEDS = {
    "Rice": 80,
    "Coffee": 60,
    "Jute": 50,
    "Cotton": 40,
    "Maize": 35,
    "Lentil": 30,
    "Mothbeans": 20,
    "Kidneybeans": 40,
    "Pigeonpeas": 30,
    "Chickpea": 30
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/fertilizer')
def fertilizer():
    return render_template('fertilizer.html')

@app.route('/update_sensors', methods=['POST'])
def update_sensors():
    global sensor_data
    try:
        data = request.get_json()
        if data:
            if "temp" in data:
                sensor_data["temp"] = float(data["temp"])
            if "humidity" in data:
                sensor_data["humidity"] = float(data["humidity"])
            if "ph" in data:
                sensor_data["ph"] = float(data["ph"])
            if "soil_moisture" in data:
                sensor_data["soil_moisture"] = float(data["soil_moisture"])
            return "OK", 200
    except:
        pass
    return "Error", 400

def get_address_details(lat, lon):
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}&zoom=14"
        headers = {'User-Agent': 'AgriTechApp/1.0'}
        resp = requests.get(url, headers=headers).json()
        addr = resp.get('address', {})
        area = addr.get('village') or addr.get('town') or addr.get('city_district') or addr.get('county') or "Unknown Area"
        state = addr.get('state', '')
        full_address = f"{area}, {state}"
        
        raw_display = resp.get('display_name', '').lower()
        if any(x in raw_display for x in ['street', 'road', 'lane', 'nagar', 'colony', 'apartment']):
            land_type = "Urban / Non-Cultivated"
        else:
            land_type = "Agricultural / Open Land"
            
        return full_address, land_type
    except:
        return "Unknown Location", "Unknown"

@app.route('/get_prediction', methods=['POST'])
def get_prediction():
    global sensor_data

    req = request.get_json()
    lat = req.get('lat')
    lon = req.get('lon')

    if lat and lon and lat != 0:
        if abs(sensor_data["last_lat"] - lat) > 0.001:
            try:
                end_date = datetime.date.today()
                start_date = end_date - datetime.timedelta(days=120)
                url = "https://archive-api.open-meteo.com/v1/archive"
                params = {
                    "latitude": lat,
                    "longitude": lon,
                    "start_date": start_date,
                    "end_date": end_date,
                    "daily": "precipitation_sum",
                    "timezone": "auto"
                }
                resp = requests.get(url, params=params).json()
                if "daily" in resp:
                    valid_rain = [x for x in resp["daily"]["precipitation_sum"] if x is not None]
                    sensor_data["seasonal_rain"] = round(sum(valid_rain), 2)
            except:
                pass

            addr, l_type = get_address_details(lat, lon)
            sensor_data["address"] = addr
            sensor_data["land_type"] = l_type
            sensor_data["last_lat"] = lat
            sensor_data["last_lon"] = lon

    model_rain = sensor_data["seasonal_rain"]
    if model_rain > 250:
        model_rain = 250.0

    prediction = "Waiting..."
    
    if model and poly and le:
        try:
            raw_data = pd.DataFrame(
                [[sensor_data['temp'], sensor_data['humidity'], sensor_data['ph'], model_rain]], 
                columns=['temperature', 'humidity', 'ph', 'rainfall']
            )
            
            poly_data = poly.transform(raw_data)
            pred_code = model.predict(poly_data)[0]
            prediction = le.inverse_transform([pred_code])[0].capitalize()
            
        except Exception as e:
            print(f"Prediction error: {e}")
            prediction = "Error"
    else:
        prediction = "Model Offline"

    alert_msg = ""
    alert_level = "normal"
    
    required_moisture = CROP_NEEDS.get(prediction, 30)
    current_moisture = sensor_data['soil_moisture']

    if prediction not in ["Waiting...", "Error", "Model Offline"]:
        if current_moisture < required_moisture:
            gap = required_moisture - current_moisture
            if gap > 20:
                alert_level = "critical"
                alert_msg = f"CRITICAL: Soil too dry for {prediction}!"
            else:
                alert_level = "warning"
                alert_msg = f"LOW WATER: {prediction} needs moisture."
        else:
            alert_msg = f"Moisture Optimal for {prediction}"

    display_data = sensor_data.copy()
    display_data["annual_rain"] = round(model_rain, 1)

    return jsonify({
        "sensors": display_data,
        "prediction": prediction,
        "alert": {
            "msg": alert_msg,
            "level": alert_level
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)