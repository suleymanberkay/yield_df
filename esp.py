import tensorflow as tf
import joblib  # joblib modülünü içe aktarın
from sklearn.pipeline import Pipeline  # Eğer model pipeline kullanıyorsa
from sklearn.neural_network import MLPRegressor

# Eğitilmiş modeli yükle
model = joblib.load('model_pipeline.joblib')

# TensorFlow Lite formatına dönüştür
converter = tf.lite.TFLiteConverter.from_saved_model("saved_model_directory")  # Model kaydedilmiş dizini girin
tflite_model = converter.convert()

# Modeli kaydet
with open("model.tflite", "wb") as f:
    f.write(tflite_model)