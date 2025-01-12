from flask import Flask, request, render_template
import pandas as pd
from joblib import load
import numpy as np

# Flask uygulaması
app = Flask(__name__)

# Modeli yükleme
model = load('model_pipeline.joblib')

@app.route('/')
def index():
    return render_template('index.html')  # Kullanıcı arayüzü (HTML)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Kullanıcıdan gelen verileri al
        average_rain_fall_mm_per_year = float(request.form['average_rain_fall_mm_per_year'])
        pesticides_tonnes = float(request.form['pesticides_tonnes'])
        avg_temp = float(request.form['avg_temp'])
        Area = request.form['Area']
        Item = request.form['Item']
        
        # Eksik özellikleri hesaplama
        pesticides_per_hectare = pesticides_tonnes / (average_rain_fall_mm_per_year / 10000 + 1e-6)
        rainfall_std = average_rain_fall_mm_per_year / avg_temp

        # Özellikleri birleştir ve DataFrame olarak oluştur
        features = pd.DataFrame({
            'average_rain_fall_mm_per_year': [average_rain_fall_mm_per_year],
            'pesticides_tonnes': [pesticides_tonnes],
            'avg_temp': [avg_temp],
            'Area': [Area],
            'Item': [Item],
            'pesticides_per_hectare': [pesticides_per_hectare],
            'rainfall_std': [rainfall_std]
        })

        # Tahmin yap
        prediction = model.predict(features)
        prediction_original = np.exp(prediction[0])  # Logaritmik dönüşüm varsa geri dönüştür

        return render_template('index.html', prediction=prediction_original)

if __name__ == "__main__":
    app.run(debug=True)
