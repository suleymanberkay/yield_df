import pandas as pd

# CSV dosyasını oku (sensörden alınan veriler)
sensor_data = pd.read_csv('sensor_data.csv')

# Veri çerçevesini kontrol et
print(sensor_data.head())
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Ön işleme adımlarını tanımlayalım (standartlaştırma ve one-hot encoding)
preprocessor = ColumnTransformer(
    transformers=[
        ('StandardScaler', StandardScaler(), ['avg_temp', 'average_rain_fall_mm_per_year']),  # Sıcaklık ve nem için standartlaştırma
        ('OneHotEncode', OneHotEncoder(), ['Area', 'Item', 'pesticides_tonnes', 'Year', 'hg/ha_yield'])  # Area ve Item için OneHotEncoding
    ],
    remainder='passthrough'  # Diğer sütunlar olduğu gibi kalacak
)

# Sensör verilerini ön işleme uygulayalım
sensor_data_processed = preprocessor.transform(sensor_data)

import pickle

# Eğitimde kullandığınız modeli ve ön işleme aracını yükleyin
with open('dtr.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('preprocessor.pkl', 'rb') as preprocess_file:
    preprocessor = pickle.load(preprocess_file)

# Sensör verisiyle tahmin yapalım
predicted_yield = model.predict(sensor_data_processed)
print(f"Predicted Yield: {predicted_yield[0]}")
