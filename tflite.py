import socket
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from joblib import dump, load

# Veri Setini Yükleme
file_path = "processed_data.csv"  # Veri dosyasının yolunu doğru bir şekilde belirttiğinizden emin olun
data = pd.read_csv(file_path)

# 1. Özellik Mühendisliği
data['log_hg_ha_yield'] = np.log1p(data['hg/ha_yield'])

# Kategorik ve Sayısal Değişkenler
categorical_features = ['Area', 'Item']
numerical_features = ['average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']

# One-Hot Encoding ve Standard Scaler
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Eğitim ve Test Setlerinin Ayrılması
X = data[categorical_features + numerical_features]
y = data['log_hg_ha_yield']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Veriyi Ön İşleme
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# 2. TensorFlow Modeli Oluşturma
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train_preprocessed.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train_preprocessed, y_train, epochs=50, batch_size=32)

# Model Performansı
y_pred = model.predict(X_test_preprocessed).flatten()
mse = np.mean((y_test - y_pred) ** 2)
print(f"Model MSE: {mse:.4f}")

# TensorFlow Modelini Kaydetme (SavedModel formatında)
saved_model_path = "saved_model_directory"

import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import numpy as np

# Modeli Sabitlemek İçin Hazırlık
input_spec = tf.TensorSpec([None, X_train_preprocessed.shape[1]], tf.float32)

# Modeli bir tf.function olarak sar
@tf.function(input_signature=[input_spec])
def model_function(input_tensor):
    return model(input_tensor)

# ConcreteFunction oluştur
concrete_function = model_function.get_concrete_function()

# Modeli Sabitleme
frozen_func = convert_variables_to_constants_v2(concrete_function)
frozen_graph_def = frozen_func.graph.as_graph_def()

# Sabitlenmiş Modeli Kaydetme
tf.io.write_graph(frozen_graph_def, ".", "frozen_model.pb", as_text=False)
print("Model başarıyla sabitlendi ve kaydedildi.")

# TensorFlow Lite Model Dönüşümü
converter = tf.lite.TFLiteConverter.from_concrete_functions([frozen_func])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_model = converter.convert()

# TensorFlow Lite Modelini Kaydetme
tflite_model_path = "model.tflite"
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)
print(f"Model TensorFlow Lite formatında başarıyla kaydedildi: {tflite_model_path}")

# TensorFlow Lite Minimal Test
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Minimal Test
test_data = np.random.rand(1, X_train_preprocessed.shape[1]).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], test_data)

try:
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    print(f"Minimal Test Tahmin Sonucu: {output}")
except Exception as e:
    print(f"Minimal test sırasında hata oluştu: {str(e)}")


from flask import Flask, request, jsonify
from joblib import load
import tensorflow as tf
import numpy as np
import pandas as pd

app = Flask(__name__)

# TensorFlow Lite Model ve Preprocessor Yükleme
tflite_model_path = "model.tflite"
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
preprocessor = load("preprocessor.joblib")

# Sabit Değerler
area = "Turkey"
item = "Rice, paddy"
pesticides = 38554.69  # Sabit bir değer
hg_ha_yield = 86899.0  # Sabit değer olarak eklendi

app = Flask(__name__)

@app.route('/api/sensor', methods=['GET', 'POST'])
def api_sensor():
    if request.method == 'GET':
        return """
        <html>
        <head><title>Prediction API</title></head>
        <body>
            <h1>Welcome to the Prediction API</h1>
            <p>This API supports POST requests for predicting crop yields based on weather data.</p>
            <h2>Usage:</h2>
            <pre>
            POST /api/sensor
            Content-Type: application/json
            Body: {
                "avg_temp": ,
                "average_rain_fall_mm_per_year":
            }
            </pre>
            <p>For testing, use tools like <a href="https://www.postman.com/" target="_blank">Postman</a> or <code>curl</code>.</p>
        </body>
        </html>
        """

    try:
        # ESP32'den gelen veriyi al
        data = request.get_json()
        avg_temp = float(data['avg_temp'])
        avg_rainfall = float(data['average_rain_fall_mm_per_year'])

        # Özellikleri bir DataFrame olarak oluştur
        features = pd.DataFrame(
            [[area, item, avg_rainfall, pesticides, avg_temp, hg_ha_yield]],
            columns=['Area', 'Item', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'hg_ha_yield']
        )

        # Özelliklerin ön işleme adımlarına sokulması
        features_preprocessed = preprocessor.transform(features)
        input_data = features_preprocessed.toarray().astype(np.float32)

        # TensorFlow Lite tahmin
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0]

        # Tahmin sonucunu JSON uyumlu bir formata dönüştürün
        predicted_yield = float(np.exp(prediction))  # NumPy float'ı standart float'a dönüştür

        return jsonify({"predicted_yield": predicted_yield}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
