import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Eğitim verinizi zaten mevcut dataset ile başlıyoruz.
df = pd.read_csv('yield_df.csv')

# Model ve preprocessor'ü yükleyin
with open('dtr.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('preprocessor.pkl', 'rb') as preprocess_file:
    preprocessor = pickle.load(preprocess_file)

# Sensör verisini alalım (örnek: sıcaklık ve nem verisi)
# Bu veriyi sensörünüzden alarak CSV dosyasına yazmış olmalısınız
sensor_data = pd.read_csv('sensor_data.csv')

sensor_data = sensor_data[['Area', 'Item', 'Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']]

sensor_data_processed = preprocessor.transform(sensor_data)

predicted_yield = model.predict(sensor_data_processed)

print(f"Predicted Yield for the given sensor data: {predicted_yield[0]}")
