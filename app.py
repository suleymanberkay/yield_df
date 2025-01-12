from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from joblib import load

# Flask uygulaması
app = Flask(__name__)

# Model ve ön işleme araçlarını yükleme
model = load('model_pipeline.joblib')

@app.route('/')
def index():
    return "IoT Tahmin Sistemi"

@app.route('/api/sensor', methods=['POST'])
def predict_from_sensor():
    data = request.get_json()

    # Sensör verileri
    avg_temp = data['avg_temp']
    average_rain_fall_mm_per_year = data['average_rain_fall_mm_per_year']
    Item = data['Item']  # Manuel olarak seçilecek
    Area = "Turkey"  # Sabit

    # Sabit değerler
    pesticides_tonnes = 38554.69

    # Ek özelliklerin hesaplanması
    pesticides_per_hectare = pesticides_tonnes / (average_rain_fall_mm_per_year / 10000 + 1e-6)
    rainfall_std = average_rain_fall_mm_per_year / avg_temp

    # Tahmin için veri oluşturma
    features = pd.DataFrame({
        'average_rain_fall_mm_per_year': [average_rain_fall_mm_per_year],
        'pesticides_tonnes': [pesticides_tonnes],
        'avg_temp': [avg_temp],
        'Area': [Area],
        'Item': [Item],
        'pesticides_per_hectare': [pesticides_per_hectare],
        'rainfall_std': [rainfall_std]
    })

    # Tahmin
    prediction = model.predict(features)
    prediction_original = np.exp(prediction[0])  # Log dönüşüm geri alınır

    return jsonify({
        "Item": Item,
        "Predicted Yield (kg/ha)": prediction_original
    })

if __name__ == "__main__":
    app.run(debug=True)
