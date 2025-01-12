import pandas as pd

# Sensör verilerini oku
sensor_data = pd.read_csv('sensor_data.csv')

# Sabit veriler (örnek)
area = 'Turkey'
items = ['Maize', 'Potatoes', 'Rice', 'Paddy', 'Sorghum', 'Soybeans', 'Wheat']
hg_ha_yield = [72614, 323303, 86899, 86899, 70968, 36869, 24400]  # Sabit verim değerleri
pesticides_tonnes = 38554.69  # Sabit pesticides değeri
average_rain_fall_mm_per_year = 593.0  # Sabit yağış miktarı

# Processed data listesi
processed_data = []
index_offset = 4697  # Başlangıç index numarası

# Sensör verilerini işleyerek modelin formatına uygun hale getirelim
for index, row in sensor_data.iterrows():
    temp = row['avg_temp']  # Sensörden alınan sıcaklık
    humidity = row['humidity'] / 10  # Nem verisini 100 ile çarpıyoruz (yüzdeyi ondalıklı sayıya dönüştür)

    # Yıl bilgisini sensör verisi üzerinden sabit tutarak işleyeceğiz
    for i, item in enumerate(items):
        # Yıl bilgisini ilerleteceğiz
        year = 2010 + (index // len(items))  # Yıl bilgisini iterasyona göre artırıyoruz
        processed_data.append({
            'Index': index_offset,  # yield_df.csv'deki indeks numarası
            'Area': area,
            'Item': item,
            'Year': year,
            'hg/ha_yield': hg_ha_yield[i],  # Sabit verim
            'average_rain_fall_mm_per_year': average_rain_fall_mm_per_year,  # Sabit yağış
            'pesticides_tonnes': pesticides_tonnes,  # Sabit pesticides
            'avg_temp': temp,  # Sensörden alınan sıcaklık
            'humidity': humidity  # Sensörden alınan nem
        })
        index_offset += 1  # Her işlemde indexi artır

# DataFrame'e dönüştür
df = pd.DataFrame(processed_data)

# Veri çerçevesini kontrol et
print(df.head())

# Yeni verileri 'yield_df.csv' dosyasına yazalım
df.to_csv('updated_yield_df.csv', index=False)
