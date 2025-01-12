import pandas as pd

# CSV dosyasını yükleyelim (dosya adını değiştirebilirsiniz)
file_path = 'yield_df.csv'  # Dosya yolunu güncelleyin
data = pd.read_csv(file_path)
# İlk olarak, aynı yıl ve aynı ürün için sıcaklıkların ortalamasını alalım
data_grouped_by_year = data.groupby(['Area', 'Item', 'Year']).agg({
    'hg/ha_yield': 'mean',
    'average_rain_fall_mm_per_year': 'mean',
    'pesticides_tonnes': 'mean',
    'avg_temp': 'mean'
}).reset_index()

# Daha sonra, aynı ülke ve ürün için tüm yılların ortalamasını alalım
final_dataset = data_grouped_by_year.groupby(['Area', 'Item']).agg({
    'hg/ha_yield': 'mean',
    'average_rain_fall_mm_per_year': 'mean',
    'pesticides_tonnes': 'mean',
    'avg_temp': 'mean'
}).reset_index()

# Sayısal değerleri sınırlandıralım (2 ondalık basamağa yuvarlayalım)
final_dataset = final_dataset.round(2)

# Sonuçları bir CSV dosyasına kaydedelim
output_file_path = 'processed_data.csv'  # Çıktı dosya yolunu güncelleyin
final_dataset.to_csv(output_file_path, index=False)

print("Veriler başarıyla işlendi ve kaydedildi!")