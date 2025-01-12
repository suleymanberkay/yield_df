import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Veri Setini Yükleme
file_path = "processed_data.csv"  # Dosya yolunu güncelleyin
data = pd.read_csv(file_path)

# 1. Yeni Özellikler Ekleyelim
data['pesticides_per_hectare'] = data['pesticides_tonnes'] / (data['hg/ha_yield'] / 10000 + 1e-6)  # Hektar başına pestisit
data['rainfall_std'] = data['average_rain_fall_mm_per_year'] / data['avg_temp']  # Yağışın sıcaklıkla normalize edilmiş standart sapması

# Logaritmik dönüşüm hedef değişken için
data['log_hg_ha_yield'] = np.log1p(data['hg/ha_yield'])

# 2. Kategorik Değişkenlerin Encode Edilmesi
categorical_features = ['Area', 'Item']
numerical_features = ['average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp',
                      'pesticides_per_hectare', 'rainfall_std']
target = 'log_hg_ha_yield'

# One-Hot Encoding ve Standard Scaling
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# 3. Eğitim ve Test Setlerinin Ayrılması
X = data[categorical_features + numerical_features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Modellerin Eğitilmesi ve Optimize Edilmesi
# Neural Network Modeli
mlp = MLPRegressor(random_state=42, max_iter=1000)
mlp_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('mlp', mlp)])

param_grid_mlp = {
    'mlp__hidden_layer_sizes': [(50, 50), (100, 50), (100, 100)],
    'mlp__activation': ['relu', 'tanh'],
    'mlp__alpha': [0.0001, 0.001, 0.01],
    'mlp__learning_rate_init': [0.001, 0.01]
}

mlp_search = GridSearchCV(mlp_pipeline, param_grid_mlp, scoring='r2', cv=3, verbose=2)
mlp_search.fit(X_train, y_train)

# Optimize Edilmiş Modelin Performansı
best_mlp_model = mlp_search.best_estimator_
y_pred_mlp = best_mlp_model.predict(X_test)
mse_mlp = mean_squared_error(y_test, y_pred_mlp)
r2_mlp = r2_score(y_test, y_pred_mlp)

print("En iyi hiperparametreler (MLP):", mlp_search.best_params_)
print("MLP Model Performansı:")
print("Mean Squared Error (MSE):", mse_mlp)
print("R2 Score:", r2_mlp)

# 5. Görselleştirme
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred_mlp, alpha=0.7, color='purple', label="Neural Network")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='gray', label="Ideal Fit")
plt.xlabel("Gerçek Değerler")
plt.ylabel("Tahmin Edilen Değerler")
plt.title("Gerçek vs Tahmin (Neural Network)")
plt.legend()
plt.show()



import numpy as np

random_index = np.random.randint(0, len(X_test))  # Test setinden rastgele bir indeks seç
sample_data = X_test.iloc[random_index:random_index + 1]
true_value = y_test.iloc[random_index]

predicted_value = best_mlp_model.predict(sample_data)

print(f"Seçilen Örnek (Index: {random_index}):")
print(sample_data)
print("\nGerçek Değer (log ölçeğinde): {:.4f}".format(true_value))
print("Tahmin Edilen Değer (log ölçeğinde): {:.4f}".format(predicted_value[0]))

# Logaritmadan geri dönüş yapma (eğer hedef değişken logaritmik dönüştürülmüşse)
true_value_original = np.exp(true_value)
predicted_value_original = np.exp(predicted_value[0])

print("\nGerçek Değer (orijinal ölçek): {:.2f}".format(true_value_original))
print("Tahmin Edilen Değer (orijinal ölçek): {:.2f}".format(predicted_value_original))


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Gerçek ve Tahmin Edilen Değerleri DataFrame'e Dönüştürme
y_test_original = np.exp(y_test)  # Logaritmadan geri dönüş
y_pred_original = np.exp(y_pred_mlp)

results_df = X_test.copy()
results_df['True Yield'] = y_test_original
results_df['Predicted Yield'] = y_pred_original

# Ülkelere Göre Verim Karşılaştırması
country_comparison = results_df.groupby('Area')[['True Yield', 'Predicted Yield']].mean()

plt.figure(figsize=(12, 6))
country_comparison.plot(kind='bar', figsize=(12, 6), width=0.8, alpha=0.9)
plt.title('Ülkelere Göre Gerçek ve Tahmin Edilen Verim Karşılaştırması')
plt.ylabel('Verim (kg/ha)')
plt.xlabel('Ülkeler')
plt.xticks(rotation=45)
plt.tight_layout()
plt.legend(loc='upper left')
plt.show()

# Ürünlere Göre Verim Karşılaştırması
item_comparison = results_df.groupby('Item')[['True Yield', 'Predicted Yield']].mean()

plt.figure(figsize=(12, 6))
item_comparison.plot(kind='bar', figsize=(12, 6), width=0.8, alpha=0.9)
plt.title('Ürünlere Göre Gerçek ve Tahmin Edilen Verim Karşılaştırması')
plt.ylabel('Verim (kg/ha)')
plt.xlabel('Ürünler')
plt.xticks(rotation=45)
plt.tight_layout()
plt.legend(loc='upper left')
plt.show()

# Artıkların hesaplanması
residuals = true_value_original - predicted_value_original

# Artıkların görselleştirilmesi
plt.figure(figsize=(8, 6))
plt.scatter(true_value_original, residuals, alpha=0.7, color='blue')
plt.axhline(y=0, color='gray', linestyle='--')
plt.xlabel("Gerçek Değer (orijinal ölçek)")
plt.ylabel("Artık (Residual)")
plt.title("Artık Analizi")
plt.show()


performance_df = sample_data.copy()
performance_df["True Value"] = true_value_original
performance_df["Predicted Value"] = predicted_value_original
performance_df["Residual"] = residuals

print("\nModel Performansı Özeti:")
print(performance_df)

import seaborn as sns
import matplotlib.pyplot as plt

# Tüm test setindeki değerlerin karşılaştırılması
plt.figure(figsize=(10, 6))
sns.kdeplot(y_test_original, label="Gerçek Değer", fill=True)  # KDE için tüm gerçek değerler
sns.kdeplot(y_pred_original, label="Tahmin Edilen Değer", fill=True)  # KDE için tüm tahmin edilen değerler

plt.title("Gerçek vs Tahmin Edilen Değerlerin Dağılımı")
plt.xlabel("Verim (kg/ha)")
plt.ylabel("Yoğunluk")
plt.legend()
plt.show()


confidence_interval = 1.96 * (y_pred_mlp.std() / np.sqrt(len(y_pred_mlp)))

print(f"\nTahminler için Güven Aralığı: ±{confidence_interval:.2f}")


from scipy.optimize import linprog
import pandas as pd
import matplotlib.pyplot as plt

# Örnek veri
optimization_data = {
    "Item": ["Maize", "Wheat", "Soybeans", "Potatoes"],
    "Predicted Yield (kg/ha)": [130000, 115000, 90000, 150000],
    "Land Required (ha)": [1, 1.2, 0.8, 1.5],
    "Water Required (liters)": [3000, 2500, 2000, 3500],
    "Pesticide Required (kg)": [2, 1.5, 1, 2.5]
}

df_optimization = pd.DataFrame(optimization_data)

# Mevcut kaynaklar
available_land = 10  # hektar
available_water = 25000  # litre
available_pesticide = 15  # kg

# Amaç fonksiyonu: Maksimum verim için negatif değerler
objective = -df_optimization["Predicted Yield (kg/ha)"].values

# Kısıtlar: Toplam alan, su ve pestisit kullanımı sınırı
A = [
    df_optimization["Land Required (ha)"].values,
    df_optimization["Water Required (liters)"].values,
    df_optimization["Pesticide Required (kg)"].values
]
b = [available_land, available_water, available_pesticide]

# Çeşitlilik kısıtı: Her ürüne en az 0.5 hektar tahsis edilmeli
min_allocation = 0.5
bounds = [(min_allocation, None) for _ in range(len(df_optimization))]

# Optimizasyonu çalıştır
result = linprog(c=objective, A_ub=A, b_ub=b, bounds=bounds, method="highs")

# Sonuçları yazdır
if result.success:
    print("Optimizasyon Başarılı!")
    print("Önerilen Ekim Planı (hektar):", result.x)
    df_optimization["Optimal Land Allocation (ha)"] = result.x
else:
    print("Optimizasyon Başarısız:", result.message)

# Çıktıların görselleştirilmesi
plt.figure(figsize=(10, 6))
plt.bar(df_optimization["Item"], df_optimization["Optimal Land Allocation (ha)"], color='green', alpha=0.7)
plt.xlabel("Ürünler")
plt.ylabel("Optimal Tahsis Edilen Alan (hektar)")
plt.title("Optimizasyon Sonuçları: Ürünlere Tahsis Edilen Alan")
plt.show()



from joblib import dump
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Örnek model ve pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Area', 'Item'])
    ]
)

mlp = MLPRegressor(random_state=42, max_iter=1000)
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('mlp', mlp)])

# Modeli kaydetme
pipeline.fit(X_train, y_train)  # Modeli eğit
dump(pipeline, 'model_pipeline.joblib')  # Modeli joblib formatında kaydet
print("Model başarıyla kaydedildi!")

from joblib import load

# Modeli yükleme
pipeline = load('model_pipeline.joblib')
print("Model başarıyla yüklendi!")
